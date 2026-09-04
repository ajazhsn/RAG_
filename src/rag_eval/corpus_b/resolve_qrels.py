"""
For each gold question, finds which chunk(s) in a given pool actually
contain that fact's raw text — this is Corpus B's qrels, built
programmatically (via substring search, restricted to the correct company's
chunks) rather than hand-labeled, since the XBRL-sourced gold_raw_text gives
us an exact string to search for.

Run once per pool (naive and table-aware) since the same fact lands in a
different chunk depending on how that pool split the document.

Usage:
  python -m rag_eval.corpus_b.resolve_qrels --pool naive
  python -m rag_eval.corpus_b.resolve_qrels --pool table_aware
"""

import argparse
import json
import re
from pathlib import Path

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.resolve_qrels")

ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "data/processed/sec10k_gold_questions.jsonl"


def has_genuine_match(raw: str, text: str) -> bool:
    """Plain substring search is unsafe for numbers — '500,000' is a literal
    substring of '31,500,000'. A match only counts if it's not immediately
    flanked by another digit or comma (i.e., it's the WHOLE number, not a
    fragment of a bigger one)."""
    for m in re.finditer(re.escape(raw), text):
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if before.isdigit() or before == "," or after.isdigit() or after == ",":
            continue
        return True
    return False


def main(pool_name: str):
    pool_path = ROOT / f"data/processed/sec10k_pool_{pool_name}.jsonl"
    out_path = ROOT / f"data/processed/sec10k_qrels_{pool_name}.jsonl"

    gold = []
    with open(GOLD_PATH) as f:
        for line in f:
            gold.append(json.loads(line))

    chunks_by_cik = {}
    with open(pool_path) as f:
        for line in f:
            chunk = json.loads(line)
            chunks_by_cik.setdefault(chunk["cik"], []).append(chunk)

    log.info(f"resolving qrels for {len(gold)} questions against pool '{pool_name}'...")

    qrels = []
    n_zero_matches = 0
    n_multi_matches = 0

    for g in gold:
        cik = g["cik"]
        raw = g["gold_raw_text"].strip()
        candidates = chunks_by_cik.get(cik, [])

        matching_ids = [c["chunk_id"] for c in candidates if has_genuine_match(raw, c["text"])]

        if len(matching_ids) == 0:
            n_zero_matches += 1
        elif len(matching_ids) > 1:
            n_multi_matches += 1

        qrels.append({
            "question_id": g["question_id"],
            "gold_chunk_ids": matching_ids,
        })

    with open(out_path, "w") as f:
        for q in qrels:
            f.write(json.dumps(q) + "\n")

    log.info(f"done -> {out_path}")
    log.info(f"questions with 0 matching chunks: {n_zero_matches} / {len(gold)}")
    log.info(f"questions with 2+ matching chunks (ambiguous): {n_multi_matches} / {len(gold)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=["naive", "table_aware"], required=True)
    args = parser.parse_args()
    main(args.pool)
