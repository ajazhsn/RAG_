"""
Runs the naive baseline end-to-end:
  question -> embed -> retrieve top-k -> generate answer (Groq, cited)
over a subset of Corpus A questions, and saves the results as JSONL —
this is what feeds the Phase 1 labeling tool.

This is deliberately the simplest possible config: one embedding model,
fixed top-k, no reranking, no query rewriting. It's the baseline every later
ablation gets compared against.

Usage:
  python -m rag_eval.pipeline.run_naive_baseline
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_eval.pipeline.embed import embed_pool, embed_query, load_config
from rag_eval.pipeline.retrieve import top_k
from rag_eval.generation.groq_client import generate_answer
from rag_eval.utils.logging import get_logger

log = get_logger("pipeline.run_naive_baseline")
load_dotenv()

ROOT = Path(__file__).resolve().parents[3]


def load_pool_lookup(processed_dir: Path) -> dict[str, dict]:
    pool_path = processed_dir / "hotpotqa_pool.jsonl"
    lookup = {}
    with open(pool_path) as f:
        for line in f:
            row = json.loads(line)
            lookup[row["paragraph_id"]] = row
    return lookup


def load_questions(processed_dir: Path, n: int) -> list[dict]:
    q_path = processed_dir / "hotpotqa_questions.jsonl"
    questions = []
    with open(q_path) as f:
        for line in f:
            questions.append(json.loads(line))
            if len(questions) >= n:
                break
    return questions


def main(n_items: int = 200):
    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]

    log.info("loading / embedding pool (cached after first run)...")
    pool_embeddings, paragraph_ids = embed_pool(cfg)
    pool_lookup = load_pool_lookup(processed_dir)

    model_key = cfg["embeddings"]["default"]
    model_hf_id = next(
        c["hf_id"] for c in cfg["embeddings"]["candidates"] if c["name"] == model_key
    )
    query_model = SentenceTransformer(model_hf_id, device="cpu")

    k = cfg["retrieval"]["top_k_candidates"][1]  # use the "5" entry as the naive default
    gen_model = cfg["generation"]["model"]

    questions = load_questions(processed_dir, n_items)
    log.info(f"running naive baseline on {len(questions)} questions, top_k={k}...")

    results = []
    for q in tqdm(questions, desc="baseline"):
        q_emb = embed_query(q["question"], query_model)
        retrieved = top_k(q_emb, pool_embeddings, paragraph_ids, k=k)

        passages = []
        for pid, score in retrieved:
            p = pool_lookup[pid]
            passages.append({"paragraph_id": pid, "title": p["title"], "text": p["text"], "score": score})

        try:
            answer = generate_answer(q["question"], passages, gen_model)
        except Exception as e:
            log.warning(f"generation failed for {q['question_id']}: {type(e).__name__}: {e}")
            answer = None

        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "retrieved_passages": passages,
            "generated_answer": answer,
        })
        time.sleep(9)  # pacing — qwen3.6-27b caps at 8000 tokens/min; ~9s between calls stays under that

    out_dir = ROOT / "data" / "processed" / "baseline_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "naive_baseline_outputs.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_failed = sum(1 for r in results if r["generated_answer"] is None)
    log.info(f"done. {len(results)} items -> {out_path}")
    log.info(f"generation failures: {n_failed} / {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="number of questions to run (default 200)")
    args = parser.parse_args()
    main(n_items=args.limit)
