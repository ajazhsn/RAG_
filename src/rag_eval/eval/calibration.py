"""
Compares the LLM judge's verdicts against your own 200 hand labels —
the actual payoff of the whole calibration exercise.

Reports, for both faithfulness and correctness:
  - Raw agreement %: how often judge and human gave the exact same rating
  - Cohen's kappa: agreement corrected for chance. Raw agreement alone can
    look deceptively high if one category dominates (e.g. if 90% of items
    are "yes", a judge that always says "yes" gets 90% raw agreement while
    being useless). Kappa accounts for this — it's the number worth quoting.

Kappa interpretation (standard Landis & Koch scale):
  <0:        worse than chance
  0.00-0.20: slight
  0.21-0.40: fair
  0.41-0.60: moderate
  0.61-0.80: substantial
  0.81-1.00: almost perfect

Also surfaces every disagreement — these are the most interesting cases to
read personally, since they reveal exactly where the judge's judgment
diverges from yours (and sometimes reveal your own labeling was the one
that was off, on reflection).
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = ROOT / "data/processed/baseline_runs/naive_baseline_outputs.jsonl"
HUMAN_LABELS_PATH = ROOT / "data/processed/labels/hand_labels.json"
JUDGE_VERDICTS_PATH = ROOT / "data/processed/labels/judge_verdicts.json"


def cohens_kappa(human_ratings: list[str], judge_ratings: list[str]) -> float:
    """Standard Cohen's kappa for two raters over categorical labels."""
    n = len(human_ratings)
    categories = sorted(set(human_ratings) | set(judge_ratings))

    po = sum(1 for h, j in zip(human_ratings, judge_ratings) if h == j) / n

    human_counts = Counter(human_ratings)
    judge_counts = Counter(judge_ratings)
    pe = sum((human_counts[c] / n) * (judge_counts[c] / n) for c in categories)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    human_labels = json.loads(HUMAN_LABELS_PATH.read_text())
    judge_verdicts = json.loads(JUDGE_VERDICTS_PATH.read_text())

    questions = {}
    with open(BASELINE_PATH) as f:
        for line in f:
            row = json.loads(line)
            questions[row["question_id"]] = row["question"]

    faith_human, faith_judge = [], []
    correct_human, correct_judge = [], []
    disagreements = []

    for qid, human in human_labels.items():
        judge = judge_verdicts.get(qid)
        if judge is None:
            continue

        faith_human.append(human["faithfulness"])
        faith_judge.append(judge["faithfulness"])
        correct_human.append(human["correctness"])
        correct_judge.append(judge["correctness"])

        if human["faithfulness"] != judge["faithfulness"] or human["correctness"] != judge["correctness"]:
            disagreements.append({
                "question_id": qid,
                "question": questions.get(qid, "?"),
                "human": {"faithfulness": human["faithfulness"], "correctness": human["correctness"]},
                "judge": {"faithfulness": judge["faithfulness"], "correctness": judge["correctness"],
                          "reasoning": judge.get("reasoning", "")},
            })

    n = len(faith_human)
    faith_agree = sum(1 for h, j in zip(faith_human, faith_judge) if h == j) / n
    correct_agree = sum(1 for h, j in zip(correct_human, correct_judge) if h == j) / n
    faith_kappa = cohens_kappa(faith_human, faith_judge)
    correct_kappa = cohens_kappa(correct_human, correct_judge)

    summary = {
        "n_compared": n,
        "faithfulness": {"raw_agreement": faith_agree, "cohens_kappa": faith_kappa},
        "correctness": {"raw_agreement": correct_agree, "cohens_kappa": correct_kappa},
        "n_disagreements": len(disagreements),
    }

    print(json.dumps(summary, indent=2))

    out_path = ROOT / "data/processed/labels/judge_calibration_report.json"
    out_path.write_text(json.dumps({"summary": summary, "disagreements": disagreements}, indent=2))
    print(f"\nfull report (incl. all disagreements) written to {out_path}")


if __name__ == "__main__":
    main()
