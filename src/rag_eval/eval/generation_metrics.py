"""
Generation-quality metrics, computed with zero API calls — pure text comparison
against the gold short answer from HotpotQA. This is the standard SQuAD-style
approach: normalize both strings (lowercase, strip punctuation/articles/extra
whitespace), then compare as exact strings and as token-overlap F1.

These are blunt instruments — semantic correctness the model expresses
differently from the gold phrasing will score low even when it's actually
right (e.g. "Otto von Bismarck" vs gold "Prussian" — both correct, zero
lexical overlap). That's precisely why the plan calls for an LLM-judge pass
too, calibrated against your own 200 hand labels — this script is the cheap,
fast, zero-cost first pass; the judge is the more expensive, more accurate
second opinion.
"""

import json
import re
import string
from pathlib import Path


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(gold: str, generated: str) -> float:
    return 1.0 if normalize_answer(gold) == normalize_answer(generated) else 0.0

def answer_contained(gold: str, generated: str) -> float:
    """Checks whether the gold answer appears as a substring inside the
    generated response. Since this system answers in full sentences rather
    than bare phrases, this is a fairer correctness signal than exact-string
    equality — 'U2 released it first [1]' should count as correct against
    gold 'U2', even though the strings aren't equal."""
    gold_norm = normalize_answer(gold)
    gen_norm = normalize_answer(generated)
    if not gold_norm:
        return 0.0
    return 1.0 if gold_norm in gen_norm else 0.0


def f1_overlap(gold: str, generated: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    gen_tokens = normalize_answer(generated).split()
    if not gold_tokens or not gen_tokens:
        return 0.0

    common = {}
    for t in gen_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        overlap += min(c, common.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(gen_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


CITATION_PATTERN = re.compile(r"\[\d+\]|【\d+】")  # handles both [1] and 【1】 styles seen from different models


def has_citation(text: str) -> bool:
    if not text:
        return False
    return bool(CITATION_PATTERN.search(text))


ABSTENTION_PHRASES = ["don't know", "do not know", "cannot answer", "not enough information"]


def is_abstention(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in ABSTENTION_PHRASES)


def score_generation(baseline_outputs_path: Path) -> dict:
    per_question = []
    with open(baseline_outputs_path) as f:
        for line in f:
            row = json.loads(line)
            gold = row["gold_answer"]
            generated = row["generated_answer"] or ""
            abstained = is_abstention(generated)

            per_question.append({
                "question_id": row["question_id"],
                "exact_match": None if abstained else exact_match(gold, generated),
                "f1": None if abstained else f1_overlap(gold, generated),
                "has_citation": has_citation(generated),
                "abstained": abstained,
                "answer_contained": None if abstained else answer_contained(gold, generated),
            })

    n = len(per_question)
    n_abstained = sum(1 for r in per_question if r["abstained"])
    answered = [r for r in per_question if not r["abstained"]]

    def avg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_questions": n,
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / n,
        "exact_match_on_answered": avg(answered, "exact_match"),
        "f1_on_answered": avg(answered, "f1"),
        "citation_rate": avg(per_question, "has_citation"),
        "answer_contained_on_answered": avg(answered, "answer_contained"),
    }
    return {"summary": summary, "per_question": per_question}


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    baseline_path = ROOT / "data/processed/baseline_runs/naive_baseline_outputs.jsonl"

    results = score_generation(baseline_path)
    print(json.dumps(results["summary"], indent=2))

    out_path = ROOT / "data/processed/baseline_runs/generation_scores.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull per-question scores written to {out_path}")
