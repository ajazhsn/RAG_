"""
Two chunking strategies over the parsed blocks:

  NAIVE: flatten every block (text + table) into one long stream per filing,
  split into fixed ~512-word windows with 15% overlap. Tables get chopped
  wherever the window boundary happens to fall — mid-row, mid-number,
  wherever. This is what most RAG tutorials do by default.

  TABLE-AWARE: text blocks get the same fixed-size treatment, but every
  table block is kept as its OWN atomic chunk, never split and never merged
  with surrounding prose. A table that's small still gets its own chunk;
  a huge table still stays whole. The hypothesis this ablation tests: does
  keeping financial tables intact improve retrieval of numeric facts,
  since a fact and its row/column context never get separated?

Word-count is used as a cheap proxy for "tokens" here (not a real BPE
tokenizer) — consistent within this project, good enough for chunking
boundaries, not claimed to be exact.

Usage:
  python -m rag_eval.corpus_b.chunk_filings
"""

import json
from pathlib import Path

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.chunk_filings")

ROOT = Path(__file__).resolve().parents[3]
BLOCKS_PATH = ROOT / "data/processed/sec10k_parsed_blocks.jsonl"

CHUNK_SIZE_WORDS = 512
OVERLAP_WORDS = int(CHUNK_SIZE_WORDS * 0.15)


def chunk_naive(blocks: list[dict]) -> list[dict]:
    full_text = "\n\n".join(b["text"] for b in blocks)
    words = full_text.split()

    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunks.append({"text": " ".join(chunk_words), "chunk_type": "mixed"})
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS
    return chunks


def chunk_table_aware(blocks: list[dict]) -> list[dict]:
    chunks = []
    buffer_words = []

    def flush_buffer():
        if not buffer_words:
            return
        start = 0
        while start < len(buffer_words):
            end = start + CHUNK_SIZE_WORDS
            chunks.append({"text": " ".join(buffer_words[start:end]), "chunk_type": "text"})
            if end >= len(buffer_words):
                break
            start = end - OVERLAP_WORDS
        buffer_words.clear()

    for b in blocks:
        if b["type"] == "table":
            flush_buffer()
            chunks.append({"text": b["text"], "chunk_type": "table"})
        else:
            buffer_words.extend(b["text"].split())
            if len(buffer_words) >= CHUNK_SIZE_WORDS * 2:
                flush_buffer()

    flush_buffer()
    return chunks


def main():
    docs = []
    with open(BLOCKS_PATH) as f:
        for line in f:
            docs.append(json.loads(line))

    log.info(f"chunking {len(docs)} filings with both strategies...")

    naive_pool = []
    table_aware_pool = []

    for doc in docs:
        cik = doc["cik"]
        source_file = doc["source_file"]
        blocks = doc["blocks"]

        naive_chunks = chunk_naive(blocks)
        for i, c in enumerate(naive_chunks):
            naive_pool.append({
                "chunk_id": f"{cik}_naive_{i}",
                "cik": cik,
                "source_file": source_file,
                "text": c["text"],
                "chunk_type": c["chunk_type"],
            })

        ta_chunks = chunk_table_aware(blocks)
        for i, c in enumerate(ta_chunks):
            table_aware_pool.append({
                "chunk_id": f"{cik}_tableaware_{i}",
                "cik": cik,
                "source_file": source_file,
                "text": c["text"],
                "chunk_type": c["chunk_type"],
            })

    naive_path = ROOT / "data/processed/sec10k_pool_naive.jsonl"
    ta_path = ROOT / "data/processed/sec10k_pool_table_aware.jsonl"

    with open(naive_path, "w") as f:
        for c in naive_pool:
            f.write(json.dumps(c) + "\n")
    with open(ta_path, "w") as f:
        for c in table_aware_pool:
            f.write(json.dumps(c) + "\n")

    log.info(f"naive pool: {len(naive_pool)} chunks -> {naive_path}")
    log.info(f"table-aware pool: {len(table_aware_pool)} chunks -> {ta_path}")

    n_table_chunks = sum(1 for c in table_aware_pool if c["chunk_type"] == "table")
    log.info(f"table-aware pool contains {n_table_chunks} atomic table chunks")


if __name__ == "__main__":
    main()
