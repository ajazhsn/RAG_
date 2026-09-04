"""
Scores Corpus B generation output. Gold answers are NUMBERS (from XBRL
facts), not text phrases, so this is deliberately different from Corpus A's
EM/F1/containment metrics — a generated answer like "approximately $216.4
million" needs to be parsed and compared to 216369000 with sensible
tolerance for rounding, not string-matched.

Correctness rule: extract every number-like token from the generated answer,
parse it (handling commas, $, %, "million"/"billion" suffixes), and check if
ANY of them is within 1% relative tolerance of the gold value. 1% allows for
minor rounding in generated prose without accepting a wrong figure.
"""

import json
import re
from pathlib import Path

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.numeric_metrics")

ROOT = Path(__file__).resolve().parents[3]

ABSTENTION_PHRASES = ["don't know", "do not know", "cannot answer", "not enough information"]

NUMBER_PATTERN = re.compile(r"[\$]?-?\(?[\d,]+\.?\d*\)?\s*(million|billion|thousand|M|B|K)?", re.IGNORECASE)

SUFFIX_MULTIPLIERS = {
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
}


def is_abstention(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in ABSTENTION_PHRASES)


def extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    numbers = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0).strip()
        if not raw or not re.search(r"\d", raw):
            continue
        suffix_match = re.search(r"(million|billion|thousand|M|B|K)$", raw, re.IGNORECASE)
        multiplier = 1
        numeric_part = raw
        if suffix_match:
            suffix = suffix_match.group(0).lower()
            multiplier = SUFFIX_MULTIPLIERS.get(suffix, 1)
            numeric_part = raw[:suffix_match.start()].strip()

        is_negative = numeric_part.strip().startswith("(") or numeric_part.strip().startswith("-")
        cleaned = re.sub(r"[\$,()]", "", numeric_part).replace("-", "").strip()
        if not cleaned or not re.match(r"^\d+\.?\d*$", cleaned):
            continue
        try:
            value = float(cleaned) * multiplier
            if is_negative:
                value = -value
            numbers.append(value)
        except ValueError:
            continue
    return numbers


def is_correct(gold_value: float, generated_answer: str, tolerance: float = 0.01) -> bool:
    if not generated_answer:
        return False
    candidates = extract_numbers(generated_answer)
    if not candidates:
        return False
    for c in candidates:
        if gold_value == 0:
            if abs(c) < 1e-6:
                return True
            continue
        relative_error = abs(c - gold_value) / abs(gold_value)
        if relative_error <= tolerance:
            return True
    return False


def score_generation_outputs(outputs_path: Path) -> dict:
    rows = []
    with open(outputs_path) as f:
        for line in f:
            rows.append(json.loads(line))

    n_total = len(rows)
    n_abstained = 0
    n_correct = 0
    n_answered = 0
    per_question = []
    per_model = {}

    for r in rows:
        answer = r["generated_answer"]
        model_used = r.get("generator_model", "unknown")  # older records predate this field
        per_model.setdefault(model_used, {"n_total": 0, "n_abstained": 0, "n_answered": 0, "n_correct": 0})
        per_model[model_used]["n_total"] += 1

        abstained = is_abstention(answer)
        correct = None
        if not abstained:
            n_answered += 1
            per_model[model_used]["n_answered"] += 1
            correct = is_correct(r["gold_value"], answer)
            if correct:
                n_correct += 1
                per_model[model_used]["n_correct"] += 1
        else:
            n_abstained += 1
            per_model[model_used]["n_abstained"] += 1

        per_question.append({
            "question_id": r["question_id"],
            "generator_model": model_used,
            "abstained": abstained,
            "correct": correct,
        })

    per_model_summary = {}
    for model_name, stats in per_model.items():
        per_model_summary[model_name] = {
            "n_total": stats["n_total"],
            "abstention_rate": stats["n_abstained"] / stats["n_total"] if stats["n_total"] else 0,
            "accuracy_on_answered": stats["n_correct"] / stats["n_answered"] if stats["n_answered"] else 0,
        }

    summary = {
        "n_total": n_total,
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / n_total if n_total else 0,
        "n_answered": n_answered,
        "n_correct": n_correct,
        "accuracy_on_answered": n_correct / n_answered if n_answered else 0,
        "per_model_breakdown": per_model_summary,
    }
    return {"summary": summary, "per_question": per_question}


if __name__ == "__main__":
    outputs_path = ROOT / "data/processed/sec10k_generation_outputs.jsonl"
    results = score_generation_outputs(outputs_path)
    print(json.dumps(results["summary"], indent=2))

    out_path = ROOT / "data/processed/sec10k_generation_scores.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull per-question scores written to {out_path}")
