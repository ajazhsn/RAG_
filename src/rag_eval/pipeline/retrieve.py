"""
Naive dense retrieval: cosine similarity (via dot product on pre-normalized
embeddings) between a query and the full pool. No index structure (HNSW/faiss)
yet — for ~19k paragraphs a brute-force matmul is fast enough on CPU and keeps
this baseline dead simple. An indexed version is a Tier-2 item for later once
we're doing repeated large-scale sweeps.
"""

import numpy as np


def top_k(query_emb: np.ndarray, pool_embeddings: np.ndarray,
          paragraph_ids: list[str], k: int = 5) -> list[tuple[str, float]]:
    """Returns [(paragraph_id, score), ...] sorted descending, length k."""
    scores = pool_embeddings @ query_emb  # [N] cosine similarities
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]  # sort just the top-k
    return [(paragraph_ids[i], float(scores[i])) for i in top_idx]
