"""
BM25 keyword-based retrieval — the classic sparse baseline every dense-retrieval
system should be compared against. Uses rank_bm25 (pure Python, no model
download, no GPU) — this is deliberately the "cheap and dumb" comparison point:
if dense retrieval can't beat BM25, the embedding model isn't earning its
complexity.
"""

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, paragraph_ids: list[str], texts: list[str]):
        self.paragraph_ids = paragraph_ids
        tokenized_corpus = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def top_k(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.paragraph_ids[i], float(scores[i])) for i in top_idx]
