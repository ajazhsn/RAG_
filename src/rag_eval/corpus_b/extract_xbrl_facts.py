"""
Extracts standardized (us-gaap:*) financial facts from Inline XBRL SEC 10-K
filings. These filings embed machine-readable tags around every financial
figure (e.g. "this number IS Total Revenue for FY2024") — this is a far more
reliable ground-truth source than generating candidate answers with an LLM
and hoping they're correct.

We deliberately skip:
  - facts with xs:nil="true" (no value present)
  - facts tied to a "dimensional" context (segmented by product line, region,
    etc.) — these require more complex question phrasing to stay unambiguous,
    so we keep only simple top-level company-wide facts for this v1 gold set
  - non-us-gaap (company-custom) concepts — not standardized, harder to
    phrase into a universally sensible question

Usage:
  python -m rag_eval.corpus_b.extract_xbrl_facts
"""

import json
import re
from pathlib import Path

from lxml import etree

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.extract_xbrl_facts")

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data/raw/sec_10k"
MANIFEST_PATH = ROOT / "data/manifest/sec_10k_manifest.json"
OUT_PATH = ROOT / "data/processed/sec10k_facts.jsonl"

NS = {
    "ix": "http://www.xbrl.org/2013/inlineXBRL",
    "xbrli": "http://www.xbrl.org/2003/instance",
    "xbrldi": "http://xbrl.org/2006/xbrldi",
}
XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"


def parse_contexts(root) -> dict:
    contexts = {}
    for ctx in root.iter("{http://www.xbrl.org/2003/instance}context"):
        ctx_id = ctx.get("id")
        period = ctx.find("xbrli:period", NS)
        if period is None:
            continue
        has_dims = ctx.find(".//xbrldi:explicitMember", NS) is not None

        instant = period.find("xbrli:instant", NS)
        if instant is not None and instant.text:
            contexts[ctx_id] = {"type": "instant", "date": instant.text.strip(), "has_dimensions": has_dims}
            continue

        start = period.find("xbrli:startDate", NS)
        end = period.find("xbrli:endDate", NS)
        if start is not None and end is not None:
            contexts[ctx_id] = {
                "type": "duration", "start": start.text.strip(), "end": end.text.strip(),
                "has_dimensions": has_dims,
            }
    return contexts


def parse_facts(root, contexts: dict) -> list[dict]:
    facts = []
    for elem in root.iter("{http://www.xbrl.org/2013/inlineXBRL}nonFraction"):
        name = elem.get("name")
        if not name or not name.lower().startswith("us-gaap:"):
            continue
        if elem.get(XSI_NIL) == "true":
            continue

        context_ref = elem.get("contextRef")
        ctx = contexts.get(context_ref)
        if ctx is None or ctx["has_dimensions"]:
            continue

        raw_text = "".join(elem.itertext()).strip()
        if not raw_text:
            continue
        cleaned = re.sub(r"[,\s]", "", raw_text)
        cleaned = cleaned.replace("(", "-").replace(")", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue

        try:
            scale = int(elem.get("scale", "0") or "0")
        except ValueError:
            scale = 0
        value *= 10 ** scale
        if elem.get("sign") == "-":
            value = -value

        concept = name.split(":", 1)[1]
        facts.append({
            "concept": concept,
            "context": ctx,
            "value": value,
            "unit": elem.get("unitRef"),
            "raw_text": raw_text,
        })
    return facts


def process_filing(html_path: Path) -> list[dict]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        tree = etree.parse(str(html_path), parser)
    except Exception as e:
        log.warning(f"failed to parse {html_path.name}: {e}")
        return []
    root = tree.getroot()
    if root is None:
        return []
    contexts = parse_contexts(root)
    facts = parse_facts(root, contexts)
    return facts


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    cik_to_company = {str(row["cik"]): row["company"] for row in manifest}

    all_records = []
    html_files = sorted(RAW_DIR.glob("*.html"))
    log.info(f"processing {len(html_files)} filings...")

    for i, html_path in enumerate(html_files, 1):
        cik = html_path.stem.split("_")[0]
        company = cik_to_company.get(cik, "UNKNOWN")

        facts = process_filing(html_path)
        for f in facts:
            all_records.append({
                "cik": cik,
                "company": company,
                "source_file": html_path.name,
                **f,
            })

        if i % 20 == 0 or i == len(html_files):
            log.info(f"[{i}/{len(html_files)}] processed, {len(all_records)} facts so far")

    with open(OUT_PATH, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    log.info(f"done. {len(all_records)} total facts extracted -> {OUT_PATH}")

    from collections import Counter
    concept_counts = Counter(r["concept"] for r in all_records)
    log.info("Top 25 most common concepts (non-dimensional, non-nil):")
    for concept, count in concept_counts.most_common(25):
        print(f"  {concept:<45} {count}")


if __name__ == "__main__":
    main()
