"""
LLM judge: rates each baseline item on the SAME two dimensions and SAME
categories (yes/partial/no) you used for hand-labeling — this identical
rubric is what makes a direct agreement comparison meaningful.

The judge is given the gold answer as a reference for correctness (this
isn't "cheating" — it's the standard way to check correctness against a
known answer, same as you had the gold answer visible while hand-labeling).

Deliberately a different model family from the generator (openai/gpt-oss-120b
here vs the generator's own model) to avoid self-preference bias — a judge
grading its own family's output tends to be too lenient toward it.

Usage:
  python -m rag_eval.eval.judge
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from rag_eval.utils.logging import get_logger

log = get_logger("eval.judge")
load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = ROOT / "data/processed/baseline_runs/naive_baseline_outputs.jsonl"
JUDGE_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are evaluating a RAG (retrieval-augmented generation) system's output.

You will be given:
- A question
- A set of numbered retrieved passages
- A generated answer (which may cite passages like [1])
- The correct gold answer, for reference

Rate the generated answer on two dimensions:

1. FAITHFULNESS: Does the generated answer rely ONLY on information in the
   retrieved passages (no invented facts, no contradictions of the passages)?
   An abstention ("I don't know") when the passages genuinely do NOT contain
   the needed information is faithful (correct, cautious behavior).
   - "yes": fully grounded in the passages, OR a correct abstention
   - "partial": mostly grounded but includes something unsupported or a minor inferred detail
   - "no": contains fabricated claims, OR contradicts a fact stated in the passages,
     OR abstains when the passages actually DO contain the needed information —
     that is a failure to use available grounded information, not a safe default.

2. CORRECTNESS: Is the generated answer actually correct, compared to the
   gold answer? Judge by meaning, not exact wording — a correct paraphrase
   counts as correct.
   - "yes": correct
   - "partial": partially correct or incomplete
   - "no": incorrect (this includes valid abstentions that don't match gold)

Respond with ONLY a JSON object, no other text:
{"faithfulness": "yes|partial|no", "correctness": "yes|partial|no", "reasoning": "one short sentence"}
"""


def format_passages(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] {p['text']}")
    return "\n\n".join(lines)


def parse_judge_response(raw: str) -> dict:
    """Judge responses should be pure JSON, but strip code fences / stray text defensively."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(RateLimitError),
)
def judge_item(question: str, gold_answer: str, passages: list[dict], generated_answer: str) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_block = format_passages(passages)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Passages:\n{context_block}\n\n"
        f"Gold answer (reference): {gold_answer}\n\n"
        f"Generated answer: {generated_answer or '(no answer generated)'}"
    )

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
        reasoning_effort="low",
    )
    raw = resp.choices[0].message.content.strip()
    return parse_judge_response(raw)


def main(limit: int = None, only_ids: list[str] = None):
    items = []
    with open(BASELINE_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    if limit:
        items = items[:limit]
    if only_ids:
        items = [it for it in items if it["question_id"] in only_ids]

    log.info(f"running judge ({JUDGE_MODEL}) on {len(items)} items...")

    out_path = ROOT / "data/processed/labels/judge_verdicts.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    n_failed = 0
    for item in tqdm(items, desc="judging"):
        try:
            verdict = judge_item(
                item["question"], item["gold_answer"],
                item["retrieved_passages"], item["generated_answer"],
            )
            results[item["question_id"]] = verdict
        except Exception as e:
            log.warning(f"judge failed for {item['question_id']}: {type(e).__name__}: {e}")
            n_failed += 1
            # only write None if there's no existing good verdict — never clobber prior success
            if results.get(item["question_id"]) is None:
                results[item["question_id"]] = None
        time.sleep(5)

    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"done. {len(items)} items processed -> {out_path}")
    log.info(f"judge failures this run: {n_failed} / {len(items)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true",
                         help="only re-run items currently marked null in judge_verdicts.json")
    parser.add_argument("--abstained-only", action="store_true",
                         help="only re-run items where the generated answer was an abstention")
    args = parser.parse_args()

    if args.retry_failed:
        out_path = ROOT / "data/processed/labels/judge_verdicts.json"
        existing = json.loads(out_path.read_text())
        failed_ids = [k for k, v in existing.items() if v is None]
        main(only_ids=failed_ids)
    elif args.abstained_only:
        abstained_ids = []
        with open(BASELINE_PATH) as f:
            for line in f:
                row = json.loads(line)
                ans = (row["generated_answer"] or "").lower()
                if "don't know" in ans or "do not know" in ans:
                    abstained_ids.append(row["question_id"])
        log.info(f"found {len(abstained_ids)} abstained items to re-judge")
        main(only_ids=abstained_ids)
    else:
        main(limit=args.limit)
