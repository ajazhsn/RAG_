"""
Runs the generation half of the pipeline on Corpus B's 243 gold questions:
retrieve top-5 (dense retrieval, table_aware pool — the configuration that
performed best at low k in the retrieval ablation), then generate a cited
answer via Groq, reusing the exact same generator/prompt as Corpus A.

Usage:
  python -m rag_eval.corpus_b.run_generation
  python -m rag_eval.corpus_b.run_generation --limit 20   # quick test first
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

from rag_eval.corpus_b.embed_pool import embed_pool, MODEL_HF_ID
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.generation.groq_client import generate_answer
from rag_eval.pipeline.embed import load_config
from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.run_generation")
load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
POOL_NAME = "table_aware"
TOP_K = 5


def load_gold(limit: int = None):
    gold = []
    with open(ROOT / "data/processed/sec10k_gold_questions.jsonl") as f:
        for line in f:
            gold.append(json.loads(line))
            if limit and len(gold) >= limit:
                break
    return gold


def load_pool_lookup():
    lookup = {}
    with open(ROOT / f"data/processed/sec10k_pool_{POOL_NAME}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lookup[row["chunk_id"]] = row
    return lookup


def main(limit: int = None, only_ids: list[str] = None, model_override: str = None):
    cfg = load_config()
    gen_model = model_override or cfg["generation"]["model"]

    gold = load_gold(limit)
    if only_ids:
        gold = [g for g in gold if g["question_id"] in only_ids]
    log.info(f"running generation on {len(gold)} Corpus B questions "
              f"(pool={POOL_NAME}, k={TOP_K}, generator={gen_model})...")

    dense_embeddings, chunk_ids = embed_pool(POOL_NAME)
    pool_lookup = load_pool_lookup()
    query_model = SentenceTransformer(MODEL_HF_ID, device="cpu")

    out_path = ROOT / "data/processed/sec10k_generation_outputs.jsonl"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                existing[row["question_id"]] = row

    n_failed = 0

    for g in tqdm(gold, desc="corpus_b generation"):
        q_emb = query_model.encode([g["question"]], convert_to_numpy=True,
                                    normalize_embeddings=True)[0].astype(np.float32)
        hits = dense_top_k(q_emb, dense_embeddings, chunk_ids, k=TOP_K)

        passages = []
        for cid, score in hits:
            chunk = pool_lookup[cid]
            passages.append({"chunk_id": cid, "text": chunk["text"], "score": score})

        try:
            answer = generate_answer(g["question"], passages, gen_model)
        except Exception as e:
            log.warning(f"generation failed for {g['question_id']}: {type(e).__name__}: {e}")
            answer = None
            n_failed += 1

        existing[g["question_id"]] = {
            "question_id": g["question_id"],
            "question": g["question"],
            "gold_value": g["gold_value"],
            "gold_raw_text": g["gold_raw_text"],
            "company": g["company"],
            "concept": g["concept"],
            "retrieved_chunks": passages,
            "generated_answer": answer,
            "generator_model": gen_model,
        }

        # write after EVERY item — never lose completed work to a later failure
        with open(out_path, "w") as f:
            for row in existing.values():
                f.write(json.dumps(row) + "\n")

        time.sleep(9)

    log.info(f"done. {len(existing)} items total -> {out_path}")
    log.info(f"generation failures this run: {n_failed} / {len(gold)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true",
                         help="only re-run items currently marked null (failed) in the output file")
    parser.add_argument("--model", type=str, default=None,
                         help="override the generator model (e.g. when the default's daily quota is exhausted)")
    args = parser.parse_args()

    if args.retry_failed:
        out_path = ROOT / "data/processed/sec10k_generation_outputs.jsonl"
        failed_ids = []
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                if row["generated_answer"] is None:
                    failed_ids.append(row["question_id"])
        log.info(f"found {len(failed_ids)} failed items to retry")
        main(only_ids=failed_ids, model_override=args.model)
    else:
        main(limit=args.limit, model_override=args.model)
