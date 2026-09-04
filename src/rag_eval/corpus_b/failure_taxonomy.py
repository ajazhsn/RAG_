"""
Categorizes every Corpus B item into one of five buckets by cross-referencing
the generated answer against whether the gold chunk was ACTUALLY in the
top-5 retrieved for that question. This separates "the model got it wrong"
into two very different root causes: retrieval never gave it a chance, vs.
retrieval succeeded but generation still failed to use it.

Categories:
  correct                        — answered correctly
  wrong_retrieval_had_it         — wrong answer, but the gold chunk WAS retrieved
                                    (a generation failure — model misread or ignored good context)
  wrong_retrieval_missed_it      — wrong answer, gold chunk was NOT retrieved
                                    (a retrieval failure — model guessed wrong, should have abstained)
  correct_abstention             — abstained, and gold chunk was genuinely NOT retrieved
                                    (the right call — nothing to work with)
  incorrect_abstention           — abstained, but gold chunk WAS retrieved
                                    (a generation failure — had the answer, didn't use it)

Usage:
  python -m rag_eval.corpus_b.failure_taxonomy
"""

import json
from pathlib import Path
from collections import Counter

from rag_eval.corpus_b.numeric_metrics import is_abstention, is_correct
from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.failure_taxonomy")

ROOT = Path(__file__).resolve().parents[3]


def load_qrels() -> dict:
    qrels = {}
    with open(ROOT / "data/processed/sec10k_qrels_table_aware.jsonl") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["question_id"]] = set(row["gold_chunk_ids"])
    return qrels


def categorize(item: dict, qrels: dict) -> str:
    qid = item["question_id"]
    answer = item["generated_answer"]
    retrieved_ids = {c["chunk_id"] for c in item["retrieved_chunks"]}
    gold_ids = qrels.get(qid, set())
    retrieval_hit = bool(retrieved_ids & gold_ids)

    if is_abstention(answer):
        return "correct_abstention" if not retrieval_hit else "incorrect_abstention"

    if is_correct(item["gold_value"], answer):
        return "correct"

    return "wrong_retrieval_had_it" if retrieval_hit else "wrong_retrieval_missed_it"


def main():
    qrels = load_qrels()

    items = []
    with open(ROOT / "data/processed/sec10k_generation_outputs.jsonl") as f:
        for line in f:
            items.append(json.loads(line))

    categorized = []
    for item in items:
        category = categorize(item, qrels)
        categorized.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "company": item["company"],
            "concept": item["concept"],
            "generated_answer": item["generated_answer"],
            "gold_value": item["gold_value"],
            "category": category,
        })

    counts = Counter(c["category"] for c in categorized)
    n = len(categorized)

    print("\n" + "=" * 70)
    print(f"Corpus B failure taxonomy — n={n} questions")
    print("-" * 70)
    for cat in ["correct", "wrong_retrieval_had_it", "wrong_retrieval_missed_it",
                "correct_abstention", "incorrect_abstention"]:
        count = counts.get(cat, 0)
        print(f"  {cat:<30} {count:>4}  ({count/n*100:.1f}%)")
    print("=" * 70)

    retrieval_fault = counts.get("wrong_retrieval_missed_it", 0) + counts.get("correct_abstention", 0)
    generation_fault = counts.get("wrong_retrieval_had_it", 0) + counts.get("incorrect_abstention", 0)
    print(f"\nOf non-correct outcomes: retrieval-attributable={retrieval_fault}, "
          f"generation-attributable={generation_fault}")

    out_path = ROOT / "data/processed/sec10k_failure_taxonomy.json"
    out_path.write_text(json.dumps({
        "summary": dict(counts),
        "retrieval_attributable": retrieval_fault,
        "generation_attributable": generation_fault,
        "items": categorized,
    }, indent=2))
    log.info(f"\nfull taxonomy written to {out_path}")


if __name__ == "__main__":
    main()
