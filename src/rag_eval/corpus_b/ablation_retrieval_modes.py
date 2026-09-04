"""
Tests whether BM25/hybrid retrieval closes the gap dense retrieval showed on
Corpus B. Hypothesis: SEC filings are heavily templated (nearly identical
language across companies — "Total Assets", "as of December 31, 2025..."),
so dense embeddings struggle to distinguish WHICH company/number is meant.
BM25's exact keyword matching (company name, specific concept terms) may be
a much stronger signal here than it was on Corpus A's topically-distinct
Wikipedia paragraphs.

Runs dense / bm25 / hybrid on BOTH pools (naive, table_aware), reusing the
same BM25Index and reciprocal_rank_fusion built for Corpus A.

Usage:
  python -m rag_eval.corpus_b.ablation_retrieval_modes
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_eval.corpus_b.embed_pool import embed_pool, MODEL_HF_ID
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.pipeline.bm25_retrieve import BM25Index
from rag_eval.pipeline.hybrid_retrieve import reciprocal_rank_fusion
from rag_eval.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k
from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.ablation_retrieval_modes")

ROOT = Path(__file__).resolve().parents[3]
RRF_K = 60


def load_gold():
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


def load_pool_texts(pool_name: str):
    chunk_ids, texts = [], []
    with open(ROOT / f"data/processed/sec10k_pool_{pool_name}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            chunk_ids.append(row["chunk_id"])
            texts.append(row["text"])
    return chunk_ids, texts


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
    model = SentenceTransformer(MODEL_HF_ID, device="cpu")

    results_table = []

    for pool_name in ["naive", "table_aware"]:
        log.info(f"=== pool: {pool_name} ===")
        qrels = load_qrels(pool_name)
        dense_embeddings, chunk_ids = embed_pool(pool_name)

        log.info("building BM25 index...")
        _, texts = load_pool_texts(pool_name)
        bm25_index = BM25Index(chunk_ids, texts)

        max_k = max(k_values)
        dense_rankings = {}
        bm25_rankings = {}

        for g in tqdm(gold, desc=f"retrieving ({pool_name})"):
            q_emb = model.encode([g["question"]], convert_to_numpy=True,
                                  normalize_embeddings=True)[0].astype(np.float32)
            dense_hits = dense_top_k(q_emb, dense_embeddings, chunk_ids, k=max_k)
            dense_rankings[g["question_id"]] = [cid for cid, score in dense_hits]

            bm25_hits = bm25_index.top_k(g["question"], k=max_k)
            bm25_rankings[g["question_id"]] = [cid for cid, score in bm25_hits]

        for k in k_values:
            dense_at_k = {qid: ids[:k] for qid, ids in dense_rankings.items()}
            m = score_run(dense_at_k, qrels)
            results_table.append({"pool": pool_name, "mode": "dense", "k": k, **m})

            bm25_at_k = {qid: ids[:k] for qid, ids in bm25_rankings.items()}
            m = score_run(bm25_at_k, qrels)
            results_table.append({"pool": pool_name, "mode": "bm25", "k": k, **m})

            hybrid_at_k = {}
            for qid in dense_rankings:
                fused = reciprocal_rank_fusion([dense_rankings[qid], bm25_rankings[qid]], rrf_k=RRF_K)
                hybrid_at_k[qid] = [cid for cid, score in fused[:k]]
            m = score_run(hybrid_at_k, qrels)
            results_table.append({"pool": pool_name, "mode": "hybrid", "k": k, **m})

    print("\n" + "=" * 100)
    print(f"Corpus B: retrieval mode comparison — n={len(gold)} questions")
    print("-" * 100)
    print(f"{'pool':<14}{'mode':<10}{'k':<6}{'recall@k':<12}{'precision@k':<14}{'mrr':<10}{'ndcg@k':<10}")
    print("-" * 100)
    for row in results_table:
        print(f"{row['pool']:<14}{row['mode']:<10}{row['k']:<6}{row['recall_at_k']:<12.4f}"
              f"{row['precision_at_k']:<14.4f}{row['mrr']:<10.4f}{row['ndcg_at_k']:<10.4f}")
    print("=" * 100)

    out_path = ROOT / "data/processed/sec10k_ablation_retrieval_modes.json"
    out_path.write_text(json.dumps({"n_questions": len(gold), "results": results_table}, indent=2))
    log.info(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
