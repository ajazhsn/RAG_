"""
HotpotQA distractor-subset builder for Corpus A.

What this does:
  1. Downloads the HotpotQA 'distractor' split via HuggingFace datasets.
  2. Takes a deterministic, seeded subset of N questions.
  3. Builds two artifacts:
     - the retrieval pool: every unique paragraph across the subset's
       "context" fields (gold + hard-negative distractors), each with a
       stable paragraph_id — this is what gets embedded and indexed.
     - the gold qrels: for each question, which paragraph_ids are the
       true supporting facts — this is what Recall@k / MRR / nDCG get
       computed against. Real IR metrics, not LLM-judged approximations.
  4. Writes both to data/processed/ as JSONL.

Usage:
  python -m rag_eval.ingest.hotpotqa
"""

import hashlib
import json
from pathlib import Path

import yaml
from datasets import load_dataset

from rag_eval.utils.logging import get_logger

log = get_logger("ingest.hotpotqa")

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def paragraph_id(title: str, text: str) -> str:
    """Stable id derived from content, so the same paragraph always gets the same id
    even if it appears across multiple questions' context lists."""
    h = hashlib.sha256(f"{title}::{text}".encode()).hexdigest()[:16]
    return f"para_{h}"


def main():
    cfg = load_config()
    corpus_a = cfg["corpus_a"]
    seed = cfg["project"]["seed"]

    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"loading {corpus_a['hf_dataset']} ({corpus_a['hf_config']} split)...")
    ds = load_dataset(
        corpus_a["hf_dataset"],
        corpus_a["hf_config"],
        split="validation",  # validation split has gold labels; train is also labeled
        trust_remote_code=True,
    )

    ds = ds.shuffle(seed=seed)
    n = min(corpus_a["n_questions"], len(ds))
    subset = ds.select(range(n))
    log.info(f"using {n} questions (of {len(ds)} available)")

    pool: dict[str, dict] = {}   # paragraph_id -> {title, text}
    qrels = []                    # one row per question: gold paragraph_ids
    questions = []

    for row in subset:
        q_id = row["id"]
        question = row["question"]
        answer = row["answer"]

        titles = row["context"]["title"]
        sentences_lists = row["context"]["sentences"]

        supporting_titles = set(row["supporting_facts"]["title"])

        gold_para_ids = []
        for title, sentences in zip(titles, sentences_lists):
            text = " ".join(sentences).strip()
            if not text:
                continue
            pid = paragraph_id(title, text)
            if pid not in pool:
                pool[pid] = {"paragraph_id": pid, "title": title, "text": text}
            if title in supporting_titles:
                gold_para_ids.append(pid)

        questions.append({"question_id": q_id, "question": question, "answer": answer})
        qrels.append({"question_id": q_id, "gold_paragraph_ids": gold_para_ids})

    pool_path = processed_dir / "hotpotqa_pool.jsonl"
    with open(pool_path, "w") as f:
        for p in pool.values():
            f.write(json.dumps(p) + "\n")

    questions_path = processed_dir / "hotpotqa_questions.jsonl"
    with open(questions_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    qrels_path = processed_dir / "hotpotqa_qrels.jsonl"
    with open(qrels_path, "w") as f:
        for r in qrels:
            f.write(json.dumps(r) + "\n")

    log.info(f"pool: {len(pool)} unique paragraphs -> {pool_path}")
    log.info(f"questions: {len(questions)} -> {questions_path}")
    log.info(f"qrels: {len(qrels)} -> {qrels_path}")


if __name__ == "__main__":
    main()
