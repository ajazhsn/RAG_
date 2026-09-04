"""
Embeds a Corpus B chunk pool (naive or table_aware) with bge-small-en-v1.5 —
the model established as the best cost/value pick from Corpus A's ablation.
Caches to disk exactly like Corpus A's embed.py, so re-runs are instant.

Usage:
  python -m rag_eval.corpus_b.embed_pool --pool naive
  python -m rag_eval.corpus_b.embed_pool --pool table_aware
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_eval.utils.logging import get_logger

log = get_logger("corpus_b.embed_pool")

ROOT = Path(__file__).resolve().parents[3]
MODEL_HF_ID = "BAAI/bge-small-en-v1.5"


def embed_pool(pool_name: str, force: bool = False):
    pool_path = ROOT / f"data/processed/sec10k_pool_{pool_name}.jsonl"
    npz_path = ROOT / f"data/processed/sec10k_embeddings_{pool_name}.npz"
    meta_path = ROOT / f"data/processed/sec10k_embeddings_{pool_name}.meta.json"

    if npz_path.exists() and meta_path.exists() and not force:
        log.info(f"loading cached embeddings for pool '{pool_name}'")
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text())
        return data["embeddings"], meta["chunk_ids"]

    chunks = []
    with open(pool_path) as f:
        for line in f:
            chunks.append(json.loads(line))
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    log.info(f"embedding {len(texts)} chunks from pool '{pool_name}' with {MODEL_HF_ID}...")
    model = SentenceTransformer(MODEL_HF_ID, device="cpu")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    np.savez_compressed(npz_path, embeddings=embeddings)
    meta_path.write_text(json.dumps({
        "model": MODEL_HF_ID, "n_chunks": len(chunk_ids), "chunk_ids": chunk_ids,
    }))
    log.info(f"embedded and cached -> {npz_path}")
    return embeddings, chunk_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=["naive", "table_aware"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    embed_pool(args.pool, force=args.force)
