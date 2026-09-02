"""
Reranker ablation: retrieve a wider dense candidate pool (top-N), rerank
those N with a cross-encoder (BAAI/bge-reranker-base), keep the top-k after
reranking. Compares against plain dense retrieval at the same k.

Cross-encoders look at the query and passage TOGETHER (full attention across
both), which is more accurate than comparing separate embeddings — but it's
also much slower, since you can't precompute passage representations ahead
of time like you can with dense embeddings. That's the real tradeoff this
ablation quantifies: quality gain vs added latency per query.

Usage:
  python -m rag_eval.eval.ablation_reranker --n 300
"""

import argparse
import json
import time
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from rag_eval.pipeline.embed import embed_pool, embed_query, load_config
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k
from rag_eval.utils.logging import get_logger

log = get_logger("eval.ablation_reranker")

ROOT = Path(__file__).resolve().parents[3]
RERANKER_MODEL = "BAAI/bge-reranker-base"


def load_pool_lookup(processed_dir: Path) -> dict:
    lookup = {}
    with open(processed_dir / "hotpotqa_pool.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lookup[row["paragraph_id"]] = row
    return lookup


def load_qrels(processed_dir: Path) -> dict:
    qrels = {}
    with open(processed_dir / "hotpotqa_qrels.jsonl") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["question_id"]] = row["gold_paragraph_ids"]
    return qrels


def load_questions(processed_dir: Path, n: int) -> list[dict]:
    questions = []
    with open(processed_dir / "hotpotqa_questions.jsonl") as f:
        for line in f:
            questions.append(json.loads(line))
            if len(questions) >= n:
                break
    return questions


def score_run(all_retrieved: dict, qrels: dict) -> dict:
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    for qid, retrieved_ids in all_retrieved.items():
        gold_ids = qrels.get(qid, [])
        if not gold_ids:
            continue
        recalls.append(recall_at_k(retrieved_ids, gold_ids))
        precisions.append(precision_at_k(retrieved_ids, gold_ids))
        mrrs.append(reciprocal_rank(retrieved_ids, gold_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, gold_ids))
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "precision_at_k": sum(precisions) / len(precisions),
        "mrr": sum(mrrs) / len(mrrs),
        "ndcg_at_k": sum(ndcgs) / len(ndcgs),
    }


def main(n_questions: int, candidate_pool_size: int, final_k: int):
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]

    pool_lookup = load_pool_lookup(processed_dir)
    qrels = load_qrels(processed_dir)
    questions = load_questions(processed_dir, n_questions)
    log.info(f"running reranker ablation on {len(questions)} questions, "
              f"candidate_pool={candidate_pool_size}, final_k={final_k}")

    log.info("loading cached dense embeddings...")
    pool_embeddings, paragraph_ids = embed_pool(cfg)
    model_key = cfg["embeddings"]["default"]
    model_hf_id = next(c["hf_id"] for c in cfg["embeddings"]["candidates"] if c["name"] == model_key)
    query_model = SentenceTransformer(model_hf_id, device="cpu")

    log.info(f"loading reranker ({RERANKER_MODEL}) — one-time download ~280MB...")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")

    dense_only_results = {}
    reranked_results = {}
    dense_latencies = []
    rerank_latencies = []

    for q in tqdm(questions, desc="dense retrieve + rerank"):
        qid = q["question_id"]

        t0 = time.perf_counter()
        q_emb = embed_query(q["question"], query_model)
        candidates = dense_top_k(q_emb, pool_embeddings, paragraph_ids, k=candidate_pool_size)
        dense_latencies.append(time.perf_counter() - t0)

        dense_only_results[qid] = [pid for pid, score in candidates[:final_k]]

        t0 = time.perf_counter()
        pairs = [(q["question"], pool_lookup[pid]["text"]) for pid, _ in candidates]
        rerank_scores = reranker.predict(pairs)
        rerank_latencies.append(time.perf_counter() - t0)

        reranked = sorted(zip([pid for pid, _ in candidates], rerank_scores), key=lambda x: -x[1])
        reranked_results[qid] = [pid for pid, score in reranked[:final_k]]

    dense_metrics = score_run(dense_only_results, qrels)
    reranked_metrics = score_run(reranked_results, qrels)

    avg_dense_latency_ms = sum(dense_latencies) / len(dense_latencies) * 1000
    avg_rerank_latency_ms = sum(rerank_latencies) / len(rerank_latencies) * 1000

    print("\n" + "=" * 80)
    print(f"Candidate pool size: {candidate_pool_size}, final k: {final_k}, n={len(questions)}")
    print("-" * 80)
    print(f"{'config':<25}{'recall@k':<12}{'precision@k':<14}{'mrr':<10}{'ndcg@k':<10}")
    print(f"{'dense only':<25}{dense_metrics['recall_at_k']:<12.4f}{dense_metrics['precision_at_k']:<14.4f}"
          f"{dense_metrics['mrr']:<10.4f}{dense_metrics['ndcg_at_k']:<10.4f}")
    print(f"{'dense + reranker':<25}{reranked_metrics['recall_at_k']:<12.4f}{reranked_metrics['precision_at_k']:<14.4f}"
          f"{reranked_metrics['mrr']:<10.4f}{reranked_metrics['ndcg_at_k']:<10.4f}")
    print("-" * 80)
    print(f"Avg dense retrieval latency:  {avg_dense_latency_ms:.1f} ms/query")
    print(f"Avg reranking latency:        {avg_rerank_latency_ms:.1f} ms/query (extra cost)")
    print(f"Total latency w/ reranker:    {avg_dense_latency_ms + avg_rerank_latency_ms:.1f} ms/query")
    print("=" * 80)

    out_path = ROOT / "data/processed/baseline_runs/ablation_reranker_results.json"
    out_path.write_text(json.dumps({
        "n_questions": len(questions),
        "candidate_pool_size": candidate_pool_size,
        "final_k": final_k,
        "dense_only": dense_metrics,
        "dense_plus_reranker": reranked_metrics,
        "avg_dense_latency_ms": avg_dense_latency_ms,
        "avg_rerank_latency_ms": avg_rerank_latency_ms,
    }, indent=2))
    log.info(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of questions to evaluate on")
    parser.add_argument("--candidate-pool", type=int, default=20, help="dense candidates fetched before reranking")
    parser.add_argument("--final-k", type=int, default=5, help="final top-k kept after reranking")
    args = parser.parse_args()
    main(n_questions=args.n, candidate_pool_size=args.candidate_pool, final_k=args.final_k)
