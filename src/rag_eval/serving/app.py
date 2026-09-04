"""
FastAPI serving layer for the Corpus A RAG system — closes the "zero
deployment proof" resume gap. Loads the embedding model and pool once at
startup (not per-request), so latency reflects real inference cost, not
model-loading overhead.

Uses the established best config from the ablations: dense retrieval,
bge-small-en-v1.5, k=5, no reranker (the ~194x latency cost measured earlier
disqualifies it for real-time serving — this endpoint IS that real-time
serving case).

Run locally:
  uvicorn rag_eval.serving.app:app --host 0.0.0.0 --port 8000
"""

import json
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from rag_eval.pipeline.embed import embed_pool, embed_query, load_config
from rag_eval.pipeline.retrieve import top_k as dense_top_k
from rag_eval.generation.groq_client import generate_answer
from rag_eval.utils.logging import get_logger

load_dotenv()
log = get_logger("serving.app")

ROOT = Path(__file__).resolve().parents[3]

app = FastAPI(title="RAG Eval API", version="1.0")

_cfg = load_config()
_pool_embeddings, _paragraph_ids = embed_pool(_cfg)
_model_key = _cfg["embeddings"]["default"]
_model_hf_id = next(c["hf_id"] for c in _cfg["embeddings"]["candidates"] if c["name"] == _model_key)
_query_model = SentenceTransformer(_model_hf_id, device="cpu")
_gen_model = _cfg["generation"]["model"]

_pool_lookup = {}
with open(ROOT / _cfg["paths"]["processed_dir"] / "hotpotqa_pool.jsonl") as f:
    for line in f:
        row = json.loads(line)
        _pool_lookup[row["paragraph_id"]] = row

log.info(f"serving ready — {len(_paragraph_ids)} passages, embedder={_model_hf_id}, generator={_gen_model}")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str | None
    retrieved_passages: list[dict]
    latency_ms: dict


@app.get("/health")
def health():
    return {"status": "ok", "n_passages": len(_paragraph_ids), "generator": _gen_model}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.perf_counter()
    q_emb = embed_query(req.question, _query_model)
    hits = dense_top_k(q_emb, _pool_embeddings, _paragraph_ids, k=req.top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    passages = []
    for pid, score in hits:
        p = _pool_lookup[pid]
        passages.append({"paragraph_id": pid, "title": p["title"], "text": p["text"], "score": score})

    t1 = time.perf_counter()
    try:
        answer = generate_answer(req.question, passages, _gen_model)
    except Exception as e:
        log.warning(f"generation failed: {type(e).__name__}: {e}")
        answer = None
    generation_ms = (time.perf_counter() - t1) * 1000

    return QueryResponse(
        answer=answer,
        retrieved_passages=passages,
        latency_ms={"retrieval": round(retrieval_ms, 1), "generation": round(generation_ms, 1)},
    )
