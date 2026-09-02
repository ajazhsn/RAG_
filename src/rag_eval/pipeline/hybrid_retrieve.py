"""
Hybrid retrieval: combines dense (embedding) and BM25 (keyword) rankings via
Reciprocal Rank Fusion (RRF) — a standard, parameter-light way to merge two
ranked lists without needing to calibrate how to weight two different score
scales (cosine similarity vs BM25 scores aren't directly comparable, but
RANKS from each list are).

RRF formula: for each document, sum 1/(rrf_k + rank) across every list it
appears in. A document that ranks well in BOTH lists gets a high combined
score; rrf_k (default 60, standard in the literature) softens the impact of
very top ranks so one list doesn't totally dominate.
"""


def reciprocal_rank_fusion(rank_lists: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """rank_lists: each is an ordered list of paragraph_ids (best first).
    Returns (paragraph_id, fused_score) sorted best-first."""
    scores = {}
    for ranked_ids in rank_lists:
        for rank, pid in enumerate(ranked_ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
