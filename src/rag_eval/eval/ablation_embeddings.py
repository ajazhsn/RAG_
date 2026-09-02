"""
Embedding model ablation: compares 3 embedding models (bge-small-en-v1.5,
bge-m3, nomic-embed-text-v1.5) on identical retrieval metrics, using their
pre-computed embedding caches — bge-small was embedded locally on CPU,
bge-m3 and nomic-embed-text were embedded on Kaggle's GPU (CPU would have
taken hours for these larger models).

Query embedding still needs to happen locally, once per model, matching
whichever model produced that cache — mixing an embedding cache from one
model with a query embedded by a different model would produce meaningless
similarity scores, so this is handled carefully via a per-model lookup.

Usage:
  python -m rag_eval.eval.ablation_embeddings --n 300
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_eval.pipeline.embed import load_config
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k
from rag_eval.utils.logging import get_logger

log = get_logger("eval.ablation_embeddings")

ROOT = Path(__file__).resolve().parents[3]


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


def load_embedding_cache(processed_dir: Path, model_name: str):
    safe_name = model_name.replace("/", "__")
    npz_path = processed_dir / f"hotpotqa_embeddings_{safe_name}.npz"
    meta_path = processed_dir / f"hotpotqa_embeddings_{safe_name}.meta.json"
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text())
    return data["embeddings"], meta["paragraph_ids"], meta["model"]


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


def main(n_questions: int, k: int):
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]

    qrels = load_qrels(processed_dir)
    questions = load_questions(processed_dir, n_questions)
    log.info(f"comparing embedding models on {len(questions)} questions, k={k}")

    candidates = cfg["embeddings"]["candidates"]
    results_table = []

    for candidate in candidates:
        model_name = candidate["name"]
        model_hf_id = candidate["hf_id"]
        log.info(f"--- {model_name} ({model_hf_id}) ---")

        try:
            pool_embeddings, paragraph_ids, cached_hf_id = load_embedding_cache(processed_dir, model_name)
        except FileNotFoundError:
            log.warning(f"no cached embeddings found for {model_name}, skipping")
            continue

        if cached_hf_id != model_hf_id:
            log.warning(f"cache model mismatch for {model_name}: cache says {cached_hf_id}, "
                        f"config says {model_hf_id} — proceeding, but verify this is intentional")

        log.info(f"loading query encoder for {model_name} (needed to embed the {len(questions)} questions)...")
        trust_remote = "nomic" in model_hf_id.lower()
        query_model = SentenceTransformer(model_hf_id, device="cpu", trust_remote_code=trust_remote)

        retrieved = {}
        for q in tqdm(questions, desc=f"querying ({model_name})"):
            q_emb = query_model.encode([q["question"]], convert_to_numpy=True,
                                        normalize_embeddings=True)[0].astype(np.float32)
            hits = dense_top_k(q_emb, pool_embeddings, paragraph_ids, k=k)
            retrieved[q["question_id"]] = [pid for pid, score in hits]

        metrics = score_run(retrieved, qrels)
        results_table.append({"model": model_name, "hf_id": model_hf_id, "dim": candidate["dim"], **metrics})

    print("\n" + "=" * 90)
    print(f"Embedding model comparison — k={k}, n={len(questions)} questions")
    print("-" * 90)
    print(f"{'model':<22}{'dim':<8}{'recall@k':<12}{'precision@k':<14}{'mrr':<10}{'ndcg@k':<10}")
    print("-" * 90)
    for row in results_table:
        print(f"{row['model']:<22}{row['dim']:<8}{row['recall_at_k']:<12.4f}"
              f"{row['precision_at_k']:<14.4f}{row['mrr']:<10.4f}{row['ndcg_at_k']:<10.4f}")
    print("=" * 90)

    out_path = ROOT / "data/processed/baseline_runs/ablation_embeddings_results.json"
    out_path.write_text(json.dumps({"n_questions": len(questions), "k": k, "results": results_table}, indent=2))
    log.info(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(n_questions=args.n, k=args.k)
