"""
Retrieval ablation: sweeps retrieval mode (dense / bm25 / hybrid) and top-k,
scores each configuration against the real qrels using the same metrics
(Recall@k, Precision@k, MRR, nDCG@k) from retrieval_metrics.py.

Runs entirely locally, no API calls, no GPU needed — BM25 is pure Python,
dense retrieval reuses the already-cached embeddings. This is the ablation
table that answers "does hybrid actually beat dense alone?" with real numbers
instead of assuming it.

By default runs on a subset of questions (fast iteration); pass --n to scale
up to the full 2000 for final reported numbers.

Usage:
  python -m rag_eval.eval.ablation_retrieval --n 500
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_eval.pipeline.embed import embed_pool, embed_query, load_config
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.pipeline.bm25_retrieve import BM25Index
from rag_eval.pipeline.hybrid_retrieve import reciprocal_rank_fusion
from rag_eval.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k
from rag_eval.utils.logging import get_logger

log = get_logger("eval.ablation_retrieval")

ROOT = Path(__file__).resolve().parents[3]


def load_pool_and_qrels(processed_dir: Path):
    pool = []
    with open(processed_dir / "hotpotqa_pool.jsonl") as f:
        for line in f:
            pool.append(json.loads(line))

    qrels = {}
    with open(processed_dir / "hotpotqa_qrels.jsonl") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["question_id"]] = row["gold_paragraph_ids"]

    return pool, qrels


def load_questions(processed_dir: Path, n: int):
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


def main(n_questions: int, k_values: list[int]):
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]

    pool, qrels = load_pool_and_qrels(processed_dir)
    questions = load_questions(processed_dir, n_questions)
    log.info(f"running ablation on {len(questions)} questions, k values: {k_values}")

    log.info("loading cached dense embeddings...")
    pool_embeddings, paragraph_ids = embed_pool(cfg)

    model_key = cfg["embeddings"]["default"]
    model_hf_id = next(c["hf_id"] for c in cfg["embeddings"]["candidates"] if c["name"] == model_key)
    query_model = SentenceTransformer(model_hf_id, device="cpu")

    log.info("building BM25 index (one-time, ~seconds)...")
    pool_texts = [p["text"] for p in pool]
    bm25_index = BM25Index(paragraph_ids, pool_texts)

    max_k = max(k_values)
    results_table = []

    dense_rankings = {}
    bm25_rankings = {}
    for q in tqdm(questions, desc="retrieving (dense + bm25)"):
        q_emb = embed_query(q["question"], query_model)
        dense_hits = dense_top_k(q_emb, pool_embeddings, paragraph_ids, k=max_k)
        dense_rankings[q["question_id"]] = [pid for pid, score in dense_hits]

        bm25_hits = bm25_index.top_k(q["question"], k=max_k)
        bm25_rankings[q["question_id"]] = [pid for pid, score in bm25_hits]

    for k in k_values:
        dense_at_k = {qid: ids[:k] for qid, ids in dense_rankings.items()}
        m = score_run(dense_at_k, qrels)
        results_table.append({"mode": "dense", "k": k, **m})

        bm25_at_k = {qid: ids[:k] for qid, ids in bm25_rankings.items()}
        m = score_run(bm25_at_k, qrels)
        results_table.append({"mode": "bm25", "k": k, **m})

        hybrid_at_k = {}
        rrf_k = cfg["retrieval"]["hybrid_rrf_k"]
        for qid in dense_rankings:
            fused = reciprocal_rank_fusion([dense_rankings[qid], bm25_rankings[qid]], rrf_k=rrf_k)
            hybrid_at_k[qid] = [pid for pid, score in fused[:k]]
        m = score_run(hybrid_at_k, qrels)
        results_table.append({"mode": "hybrid", "k": k, **m})

    print("\n" + "=" * 90)
    print(f"{'mode':<10}{'k':<6}{'recall@k':<12}{'precision@k':<14}{'mrr':<10}{'ndcg@k':<10}")
    print("-" * 90)
    for row in results_table:
        print(f"{row['mode']:<10}{row['k']:<6}{row['recall_at_k']:<12.4f}"
              f"{row['precision_at_k']:<14.4f}{row['mrr']:<10.4f}{row['ndcg_at_k']:<10.4f}")
    print("=" * 90)

    out_path = ROOT / "data/processed/baseline_runs/ablation_retrieval_results.json"
    out_path.write_text(json.dumps({"n_questions": len(questions), "results": results_table}, indent=2))
    log.info(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of questions to evaluate on")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 10, 20])
    args = parser.parse_args()
    main(n_questions=args.n, k_values=args.k_values)
