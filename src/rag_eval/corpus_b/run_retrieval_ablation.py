"""
Runs retrieval over both Corpus B chunk pools (naive vs table-aware) and
scores each against its own qrels — this is the direct answer to "does
keeping financial tables intact actually help retrieval?"

Reuses the exact same metric functions from Corpus A's retrieval_metrics.py,
so the numbers are directly comparable in methodology (not necessarily in
absolute value, since the corpora and question styles differ).

Usage:
  python -m rag_eval.corpus_b.run_retrieval_ablation
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_eval.corpus_b.embed_pool import embed_pool, MODEL_HF_ID
from rag_eval.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.run_retrieval_ablation")

ROOT = Path(__file__).resolve().parents[3]


def load_gold() -> list[dict]:
    gold = []
    with open(ROOT / "data/processed/sec10k_gold_questions.jsonl") as f:
        for line in f:
            gold.append(json.loads(line))
    return gold


def load_qrels(pool_name: str) -> dict:
    qrels = {}
    with open(ROOT / f"data/processed/sec10k_qrels_{pool_name}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["question_id"]] = row["gold_chunk_ids"]
    return qrels


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


def main(k_values: list[int] = None):
    if k_values is None:
        k_values = [3, 5, 10, 20]

    gold = load_gold()
    log.info(f"loading query encoder ({MODEL_HF_ID})...")
    model = SentenceTransformer(MODEL_HF_ID, device="cpu")

    results_table = []

    for pool_name in ["naive", "table_aware"]:
        log.info(f"--- pool: {pool_name} ---")
        embeddings, chunk_ids = embed_pool(pool_name)
        qrels = load_qrels(pool_name)

        max_k = max(k_values)
        rankings = {}
        for g in tqdm(gold, desc=f"querying ({pool_name})"):
            q_emb = model.encode([g["question"]], convert_to_numpy=True,
                                  normalize_embeddings=True)[0].astype(np.float32)
            hits = dense_top_k(q_emb, embeddings, chunk_ids, k=max_k)
            rankings[g["question_id"]] = [cid for cid, score in hits]

        for k in k_values:
            at_k = {qid: ids[:k] for qid, ids in rankings.items()}
            m = score_run(at_k, qrels)
            results_table.append({"pool": pool_name, "k": k, **m})

    print("\n" + "=" * 90)
    print(f"Corpus B retrieval: naive vs table-aware chunking — n={len(gold)} questions")
    print("-" * 90)
    print(f"{'pool':<14}{'k':<6}{'recall@k':<12}{'precision@k':<14}{'mrr':<10}{'ndcg@k':<10}")
    print("-" * 90)
    for row in results_table:
        print(f"{row['pool']:<14}{row['k']:<6}{row['recall_at_k']:<12.4f}"
              f"{row['precision_at_k']:<14.4f}{row['mrr']:<10.4f}{row['ndcg_at_k']:<10.4f}")
    print("=" * 90)

    out_path = ROOT / "data/processed/sec10k_ablation_chunking_results.json"
    out_path.write_text(json.dumps({"n_questions": len(gold), "results": results_table}, indent=2))
    log.info(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
