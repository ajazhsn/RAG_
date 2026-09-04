"""
Parses a 10-K's raw HTML into an ordered sequence of "blocks" — paragraphs
and tables, in document order — stripping XBRL tags, scripts, and styling.

Deliberately does NOT try to classify "real financial table" vs "layout
table" up front (SEC filings mix both constantly, e.g. the cover page uses
<table> purely for positioning text). Instead every table becomes a block
tagged 'table' with its raw row/cell content preserved; the CHUNKING step
(not this one) decides whether to treat tables specially. This keeps this
parser simple and lets the naive-vs-table-aware chunking ablation be where
that decision actually gets tested with real numbers.

Usage:
  python -m rag_eval.corpus_b.parse_filings
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.parse_filings")

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data/raw/sec_10k"
OUT_PATH = ROOT / "data/processed/sec10k_parsed_blocks.jsonl"


def table_to_text(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def is_mostly_numeric(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    numeric_like = sum(1 for t in tokens if re.match(r"^[\$\(\)\-,.\d%]+$", t))
    return numeric_like / len(tokens) > 0.3


def parse_filing(html_path: Path) -> list[dict]:
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()

    blocks = []
    seen_tables = set()

    for elem in soup.find_all(["p", "table"]):
        if elem.name == "table":
            if id(elem) in seen_tables:
                continue
            for nested in elem.find_all("table"):
                seen_tables.add(id(nested))
            text = table_to_text(elem)
            if not text.strip():
                continue
            blocks.append({
                "type": "table",
                "text": text,
                "is_numeric": is_mostly_numeric(text),
            })
        else:
            # skip paragraphs that live INSIDE a table — already captured via table_to_text
            if elem.find_parent("table") is not None:
                continue
            text = elem.get_text(" ", strip=True)
            if not text or len(text) < 3:
                continue
            blocks.append({"type": "text", "text": text, "is_numeric": False})

    deduped = []
    for b in blocks:
        if deduped and deduped[-1]["text"] == b["text"]:
            continue
        deduped.append(b)

    return deduped


def main():
    html_files = sorted(RAW_DIR.glob("*.html"))
    log.info(f"parsing {len(html_files)} filings into ordered blocks...")

    all_docs = []
    total_blocks = 0
    for i, html_path in enumerate(html_files, 1):
        cik = html_path.stem.split("_")[0]
        try:
            blocks = parse_filing(html_path)
        except Exception as e:
            log.warning(f"failed to parse {html_path.name}: {e}")
            continue

        all_docs.append({
            "cik": cik,
            "source_file": html_path.name,
            "blocks": blocks,
        })
        total_blocks += len(blocks)

        if i % 20 == 0 or i == len(html_files):
            log.info(f"[{i}/{len(html_files)}] parsed, {total_blocks} blocks so far")

    with open(OUT_PATH, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")

    log.info(f"done. {len(all_docs)} filings, {total_blocks} total blocks -> {OUT_PATH}")

    n_table_blocks = sum(1 for d in all_docs for b in d["blocks"] if b["type"] == "table")
    n_numeric_tables = sum(1 for d in all_docs for b in d["blocks"] if b["type"] == "table" and b["is_numeric"])
    log.info(f"table blocks: {n_table_blocks} total, {n_numeric_tables} flagged as numeric/financial")


if __name__ == "__main__":
    main()
