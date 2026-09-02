"""
SEC EDGAR ingestion for Corpus B.

What this does:
  1. Pulls the official company ticker->CIK mapping from SEC.
  2. Picks N companies (deterministic, seeded — reproducible sample).
  3. For each, queries the submissions API for 10-K filings in the configured
     date window, and grabs the accession number + primary document.
  4. Downloads the primary document (HTML) for each filing.
  5. Writes a manifest CSV: cik, company, accession_number, filing_date,
     primary_doc_url, local_path, sha256 — this is the provenance record.
     Anyone (including an interviewer) can take a row from this manifest and
     verify it against sec.gov directly.

Usage:
  python -m rag_eval.ingest.edgar

Respects SEC's fair-access policy: requires a real contact User-Agent
(see config/settings.yaml -> corpus_b.user_agent, or set EDGAR_USER_AGENT in .env)
and rate-limits requests via request_delay_seconds.
"""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

from rag_eval.utils.logging import get_logger

log = get_logger("ingest.edgar")

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]  # repo root
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_user_agent(cfg: dict) -> str:
    ua = os.getenv("EDGAR_USER_AGENT") or cfg["corpus_b"]["user_agent"]
    if "REPLACE_ME" in ua:
        raise ValueError(
            "Set a real contact string in .env as EDGAR_USER_AGENT "
            "(e.g. 'Ajaz Hussain ajaz.research@example.com') before running this script. "
            "SEC requires a genuine User-Agent identifying the requester — see "
            "https://www.sec.gov/os/webmaster-faq#developers"
        )
    return ua


def fetch_json(url: str, headers: dict, delay: float) -> dict:
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)
    return resp.json()


def get_company_universe(headers: dict, delay: float) -> list[dict]:
    """Official SEC ticker -> CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    data = fetch_json(url, headers, delay)
    # data is a dict of index -> {cik_str, ticker, title}
    return list(data.values())


def get_filings_for_company(cik: int, headers: dict, delay: float,
                             filing_type: str, date_from: str, date_to: str) -> list[dict]:
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        data = fetch_json(url, headers, delay)
    except requests.HTTPError as e:
        log.warning(f"skip CIK {cik_padded}: {e}")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for form, fdate, acc, pdoc in zip(forms, dates, accessions, primary_docs):
        if form != filing_type:
            continue
        if not (date_from <= fdate <= date_to):
            continue
        results.append({
            "cik": cik,
            "accession_number": acc,
            "filing_date": fdate,
            "primary_document": pdoc,
        })
    return results


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_filing(cik: int, accession: str, primary_doc: str,
                     headers: dict, delay: float, out_dir: Path) -> Path | None:
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_doc}"
    out_path = out_dir / f"{cik}_{acc_nodash}.html"

    if out_path.exists():
        return out_path  # already downloaded, don't re-fetch

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        log.warning(f"failed download ({resp.status_code}): {url}")
        return None

    out_path.write_bytes(resp.content)
    time.sleep(delay)
    return out_path


def main():
    cfg = load_config()
    corpus_b = cfg["corpus_b"]
    ua = get_user_agent(cfg)
    headers = {"User-Agent": ua}
    delay = corpus_b["request_delay_seconds"]

    raw_dir = ROOT / cfg["paths"]["raw_dir"] / "sec_10k"
    manifest_dir = ROOT / cfg["paths"]["manifest_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching company universe from SEC...")
    universe = get_company_universe(headers, delay)
    log.info(f"{len(universe)} companies in SEC ticker list")

    random.seed(cfg["project"]["seed"])
    random.shuffle(universe)

    target_n = corpus_b["n_filings"]
    manifest_rows = []
    companies_tried = 0

    for company in universe:
        if len(manifest_rows) >= target_n:
            break
        companies_tried += 1
        cik = company["cik_str"]
        title = company["title"]

        filings = get_filings_for_company(
            cik, headers, delay,
            filing_type=corpus_b["filing_type"],
            date_from=corpus_b["date_from"],
            date_to=corpus_b["date_to"],
        )
        if not filings:
            continue

        # take the most recent matching filing for this company
        filing = sorted(filings, key=lambda f: f["filing_date"], reverse=True)[0]

        local_path = download_filing(
            cik, filing["accession_number"], filing["primary_document"],
            headers, delay, raw_dir,
        )
        if local_path is None:
            continue

        manifest_rows.append({
            "cik": cik,
            "company": title,
            "accession_number": filing["accession_number"],
            "filing_date": filing["filing_date"],
            "primary_document_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{filing['accession_number'].replace('-', '')}/{filing['primary_document']}"
            ),
            "local_path": str(local_path.relative_to(ROOT)),
            "sha256": sha256_of_file(local_path),
        })
        log.info(f"[{len(manifest_rows)}/{target_n}] {title} — {filing['filing_date']}")

    manifest_path = manifest_dir / "sec_10k_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2))
    log.info(
        f"done. {len(manifest_rows)} filings downloaded "
        f"(checked {companies_tried} companies). manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
