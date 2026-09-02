"""
Thin wrapper around the Groq chat completion API.
Enforces a citation-required prompt format: the generator must reference which
retrieved passage (by number) supports each claim, and must say "I don't know"
rather than guessing if the passages don't answer the question. This is what
makes citation-precision/recall and abstention-accuracy measurable later —
without a citation requirement in the prompt, there's nothing to score.
"""

import os
import re
import time

from groq import Groq
from groq import RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag_eval.utils.logging import get_logger

log = get_logger("generation.groq_client")

SYSTEM_PROMPT = """You are a question-answering assistant. You will be given a \
question and a set of numbered passages. Answer using ONLY information in the \
passages.

Rules:
- Cite the passage number(s) that support each part of your answer, like [1] or [2][3].
- If the passages do not contain enough information to answer, respond exactly: \
"I don't know based on the given information." Do not guess.
- Keep the answer concise — one or two sentences.
"""


def format_passages(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] {p['text']}")
    return "\n\n".join(lines)


def strip_reasoning(text: str) -> str:
    """Reasoning models (qwen3.6, etc.) emit a <think>...</think> block before
    the real answer. We only want the final answer for scoring/display —
    the reasoning trace itself isn't part of the citeable output."""
    if text is None:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(RateLimitError),
)
def generate_answer(question: str, passages: list[dict], model: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_block = format_passages(passages)
    user_prompt = f"Passages:\n{context_block}\n\nQuestion: {question}"

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,  # answer is meant to be 1-2 sentences — reasoning is disabled, no need for more
    )
    # qwen3.x models support disabling reasoning entirely via reasoning_effort="none" —
    # we don't need step-by-step thinking for extractive QA, and skipping it is both
    # faster (no thinking tokens burned) and avoids leaked/truncated <think> blocks.
    if model.startswith("qwen/"):
        kwargs["reasoning_effort"] = "none"
    elif model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content.strip()
    return strip_reasoning(raw)