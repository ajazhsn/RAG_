"""
Retrieval metrics, computed the exact way discussed in planning:
for each question, compare the set of retrieved paragraph_ids against the
gold_paragraph_ids from qrels. No LLM involved here — this is pure set/rank
arithmetic against ground truth, which is why it's trustworthy on its own.

Metrics:
  - Recall@k: fraction of gold paragraphs that appear in the top-k retrieved
  - Precision@k: fraction of the top-k retrieved that are actually gold
  - MRR (Mean Reciprocal Rank): 1/rank of the FIRST gold paragraph found
    (0 if none found) — rewards finding a gold paragraph early
  - nDCG@k: like MRR but accounts for ALL gold paragraphs found and their
    positions, not just the first — the standard IR ranking-quality metric
"""

import json
import math
from pathlib import Path


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return None  # undefined — shouldn't happen with HotpotQA, but guard anyway
    hit = len(set(retrieved_ids) & set(gold_ids))
    return hit / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not retrieved_ids:
        return None
    hit = len(set(retrieved_ids) & set(gold_ids))
    return hit / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    gold_set = set(gold_ids)
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    gold_set = set(gold_ids)
    dcg = 0.0
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gold_set:
            dcg += 1.0 / math.log2(rank + 1)

    # ideal DCG: as if all gold docs were ranked first (up to k slots)
    n_ideal = min(len(gold_set), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def score_retrieval(baseline_outputs_path: Path, qrels_path: Path) -> dict:
    """Loads baseline outputs + qrels, computes all retrieval metrics,
    returns per-question rows plus averaged summary."""
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            row = json.loads(line)
            qrels[row["question_id"]] = row["gold_paragraph_ids"]

    per_question = []
    with open(baseline_outputs_path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["question_id"]
            gold_ids = qrels.get(qid, [])
            retrieved_ids = [p["paragraph_id"] for p in row["retrieved_passages"]]

            per_question.append({
                "question_id": qid,
                "recall": recall_at_k(retrieved_ids, gold_ids),
                "precision": precision_at_k(retrieved_ids, gold_ids),
                "mrr": reciprocal_rank(retrieved_ids, gold_ids),
                "ndcg": ndcg_at_k(retrieved_ids, gold_ids),
            })

    def avg(key):
        vals = [r[key] for r in per_question if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_questions": len(per_question),
        "recall_at_k": avg("recall"),
        "precision_at_k": avg("precision"),
        "mrr": avg("mrr"),
        "ndcg_at_k": avg("ndcg"),
    }
    return {"summary": summary, "per_question": per_question}


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    baseline_path = ROOT / "data/processed/baseline_runs/naive_baseline_outputs.jsonl"
    qrels_path = ROOT / "data/processed/hotpotqa_qrels.jsonl"

    results = score_retrieval(baseline_path, qrels_path)
    print(json.dumps(results["summary"], indent=2))

    out_path = ROOT / "data/processed/baseline_runs/retrieval_scores.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull per-question scores written to {out_path}")
