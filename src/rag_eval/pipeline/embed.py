"""
Embeds the Corpus A paragraph pool with the configured default embedding model,
and caches the result to disk so repeated runs (or later ablations) don't
re-embed 19k paragraphs every time.

Cache format: a .npz with 'embeddings' (float32 array) and 'paragraph_ids'
(matching order), plus a small .json sidecar recording which model produced it —
so if you switch embedding models later, this cache is correctly invalidated.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from rag_eval.utils.logging import get_logger

log = get_logger("pipeline.embed")

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_pool(processed_dir: Path) -> list[dict]:
    pool_path = processed_dir / "hotpotqa_pool.jsonl"
    pool = []
    with open(pool_path) as f:
        for line in f:
            pool.append(json.loads(line))
    return pool


def get_cache_paths(processed_dir: Path, model_name: str) -> tuple[Path, Path]:
    safe_name = model_name.replace("/", "__")
    npz_path = processed_dir / f"hotpotqa_embeddings_{safe_name}.npz"
    meta_path = processed_dir / f"hotpotqa_embeddings_{safe_name}.meta.json"
    return npz_path, meta_path


def embed_pool(cfg: dict, force: bool = False) -> tuple[np.ndarray, list[str]]:
    """Returns (embeddings [N, dim], paragraph_ids [N]) — same order."""
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    model_key = cfg["embeddings"]["default"]
    model_hf_id = next(
        c["hf_id"] for c in cfg["embeddings"]["candidates"] if c["name"] == model_key
    )

    npz_path, meta_path = get_cache_paths(processed_dir, model_key)

    if npz_path.exists() and meta_path.exists() and not force:
        log.info(f"loading cached embeddings from {npz_path}")
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text())
        return data["embeddings"], meta["paragraph_ids"]

    log.info(f"embedding pool with {model_hf_id} (this runs once, then caches)...")
    pool = load_pool(processed_dir)
    paragraph_ids = [p["paragraph_id"] for p in pool]
    texts = [p["text"] for p in pool]

    model = SentenceTransformer(model_hf_id, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=cfg["embeddings"]["batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity = dot product
    ).astype(np.float32)

    np.savez_compressed(npz_path, embeddings=embeddings)
    meta_path.write_text(json.dumps({
        "model": model_hf_id,
        "n_paragraphs": len(paragraph_ids),
        "paragraph_ids": paragraph_ids,
    }))
    log.info(f"embedded {len(paragraph_ids)} paragraphs -> {npz_path}")
    return embeddings, paragraph_ids


def embed_query(text: str, model: SentenceTransformer) -> np.ndarray:
    """Takes an already-loaded model — don't reload per query, that's wasteful."""
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return emb[0].astype(np.float32)


if __name__ == "__main__":
    cfg = load_config()
    embed_pool(cfg)
