"""
Generates gold (question, answer) pairs from the curated set of high-coverage
XBRL facts — this is Corpus B's equivalent of HotpotQA's qrels: a benchmark
with objectively correct answers, sourced directly from each filing's own
structured data rather than LLM-generated guesses.

Approach:
  1. Restrict to 14 curated us-gaap concepts that appear in 70+ of our ~99
     companies (picked from the frequency analysis — excludes fund-specific
     concepts like InvestmentOwnedAtFairValue that only a handful of filers use).
  2. Pick the ~20 companies with the BEST coverage across those 14 concepts,
     so each company yields a rich, mostly-complete question set rather than
     sparse partial coverage.
  3. For each company+concept, take the fact from the MOST RECENT period
     (avoids duplicate near-identical questions across multiple fiscal years).
  4. Phrase each fact into a simple, consistent question template.

Usage:
  python -m rag_eval.corpus_b.generate_gold_questions
"""

import json
import re
from datetime import date
from pathlib import Path

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.generate_gold_questions")

ROOT = Path(__file__).resolve().parents[3]
FACTS_PATH = ROOT / "data/processed/sec10k_facts.jsonl"
OUT_PATH = ROOT / "data/processed/sec10k_gold_questions.jsonl"

N_COMPANIES = 20
MIN_SIGNIFICANT_DIGITS = 4  # values shorter than this (e.g. "10", "42") are too
                             # generic to be a meaningful, unique retrieval target —
                             # they'll spuriously match dozens of unrelated locations
                             # in a large document regardless of matching logic

CONCEPT_TEMPLATES = {
    "Assets": "What was {company}'s total assets as of {date}?",
    "NetCashProvidedByUsedInOperatingActivities":
        "What was {company}'s net cash provided by (or used in) operating activities for the period from {start} to {end}?",
    "NetCashProvidedByUsedInFinancingActivities":
        "What was {company}'s net cash provided by (or used in) financing activities for the period from {start} to {end}?",
    "NetIncomeLoss": "What was {company}'s net income (or loss) for the period from {start} to {end}?",
    "StockholdersEquity": "What was {company}'s total stockholders' equity as of {date}?",
    "NetCashProvidedByUsedInInvestingActivities":
        "What was {company}'s net cash provided by (or used in) investing activities for the period from {start} to {end}?",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents":
        "What was {company}'s total cash, cash equivalents, and restricted cash as of {date}?",
    "WeightedAverageNumberOfSharesOutstandingBasic":
        "What was {company}'s weighted average basic shares outstanding for the period from {start} to {end}?",
    "EarningsPerShareBasic": "What was {company}'s basic earnings per share for the period from {start} to {end}?",
    "WeightedAverageNumberOfDilutedSharesOutstanding":
        "What was {company}'s weighted average diluted shares outstanding for the period from {start} to {end}?",
    "EarningsPerShareDiluted": "What was {company}'s diluted earnings per share for the period from {start} to {end}?",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
        "What was {company}'s income from continuing operations before income taxes for the period from {start} to {end}?",
    "OperatingIncomeLoss": "What was {company}'s operating income (or loss) for the period from {start} to {end}?",
    "IncomeTaxExpenseBenefit": "What was {company}'s income tax expense (or benefit) for the period from {start} to {end}?",
}

CURATED_CONCEPTS = list(CONCEPT_TEMPLATES.keys())


def parse_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def period_end(ctx: dict) -> date:
    if ctx["type"] == "instant":
        return parse_date(ctx["date"])
    return parse_date(ctx["end"])


def format_question(concept: str, company: str, ctx: dict) -> str:
    template = CONCEPT_TEMPLATES[concept]
    if ctx["type"] == "instant":
        return template.format(company=company, date=ctx["date"])
    return template.format(company=company, start=ctx["start"], end=ctx["end"])


def main():
    records = []
    with open(FACTS_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    curated = [r for r in records if r["concept"] in CURATED_CONCEPTS]
    log.info(f"{len(curated)} facts match curated concept list (of {len(records)} total)")

    before_digit_filter = len(curated)
    curated = [r for r in curated if len(re.sub(r"[^0-9]", "", r["raw_text"])) >= MIN_SIGNIFICANT_DIGITS]
    log.info(f"{before_digit_filter - len(curated)} facts dropped for being too short/generic "
              f"(<{MIN_SIGNIFICANT_DIGITS} significant digits) — {len(curated)} remain")

    by_company = {}
    for r in curated:
        cik = r["cik"]
        concept = r["concept"]
        by_company.setdefault(cik, {})
        existing = by_company[cik].get(concept)
        if existing is None or period_end(r["context"]) > period_end(existing["context"]):
            by_company[cik][concept] = r

    coverage = [(cik, len(concepts)) for cik, concepts in by_company.items()]
    coverage.sort(key=lambda x: -x[1])
    top_companies = [cik for cik, _ in coverage[:N_COMPANIES]]

    log.info(f"selected {len(top_companies)} companies with best concept coverage")
    log.info(f"coverage range: {coverage[0][1]} down to {coverage[len(top_companies)-1][1]} of {len(CURATED_CONCEPTS)} concepts")

    gold_items = []
    for cik in top_companies:
        concepts = by_company[cik]
        for concept, fact in concepts.items():
            question_id = f"secqa_{cik}_{concept}"
            question = format_question(concept, fact["company"], fact["context"])
            gold_items.append({
                "question_id": question_id,
                "cik": cik,
                "company": fact["company"],
                "concept": concept,
                "question": question,
                "gold_value": fact["value"],
                "gold_raw_text": fact["raw_text"],
                "unit": fact["unit"],
                "context": fact["context"],
                "source_file": fact["source_file"],
            })

    with open(OUT_PATH, "w") as f:
        for item in gold_items:
            f.write(json.dumps(item) + "\n")

    log.info(f"generated {len(gold_items)} gold questions -> {OUT_PATH}")

    log.info("Sample questions:")
    for item in gold_items[:5]:
        print(f"  Q: {item['question']}")
        print(f"     A: {item['gold_value']:,.0f} (source: {item['source_file']})")
        print()


if __name__ == "__main__":
    main()
