# rag-eval

RAG system with a first-class evaluation harness — built eval-first, corpus-first,
so every architectural decision produces a measured row in an ablation table
rather than an unverified claim.

**→ See [REPORT.md](REPORT.md) for full methodology, every ablation result, the
judge-calibration process, the Corpus B failure taxonomy, and resume bullets.**

## Two corpora, deliberately different roles

- **Corpus A — HotpotQA (distractor split).** Pre-labeled gold supporting facts.
  This is where real IR metrics live: Recall@k, MRR, nDCG against actual qrels,
  not LLM-judged approximations. All ablation sweeps run here.
- **Corpus B — SEC 10-K filings (EDGAR).** Messy real PDFs/HTML, multi-page
  tables, no pre-existing labels. Gold Q&A pairs are built from Inline-XBRL
  structured facts (not LLM-generated), giving provenance-verifiable ground
  truth. Used for chunking ablations and a generalization test.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-embed.txt   # embedding models + Groq client
cp .env.example .env
# edit .env: fill in GROQ_API_KEY, EDGAR_USER_AGENT
```

## Reproducing the pipeline

**Corpus A:**
```bash
python -m rag_eval.ingest.hotpotqa
python -m rag_eval.pipeline.run_naive_baseline
python -m rag_eval.eval.retrieval_metrics
python -m rag_eval.eval.generation_metrics
python -m rag_eval.eval.ablation_retrieval --n 2000
python -m rag_eval.eval.ablation_reranker --n 300
python -m rag_eval.eval.ablation_embeddings --n 300   # requires bge-m3/nomic embeddings (see REPORT.md — embedded on Kaggle GPU)
```

**Judge calibration:**
```bash
python -m rag_eval.labeling.app          # hand-label 200 items at localhost:5000
python -m rag_eval.eval.judge
python -m rag_eval.eval.calibration
```

**Corpus B:**
```bash
python -m rag_eval.ingest.edgar
python -m rag_eval.corpus_b.extract_xbrl_facts
python -m rag_eval.corpus_b.generate_gold_questions
python -m rag_eval.corpus_b.parse_filings
python -m rag_eval.corpus_b.chunk_filings
python -m rag_eval.corpus_b.resolve_qrels --pool naive
python -m rag_eval.corpus_b.resolve_qrels --pool table_aware
python -m rag_eval.corpus_b.run_retrieval_ablation
python -m rag_eval.corpus_b.ablation_retrieval_modes
python -m rag_eval.corpus_b.run_generation
python -m rag_eval.corpus_b.numeric_metrics
python -m rag_eval.corpus_b.failure_taxonomy
```

**Serving:**
```bash
pip install fastapi "uvicorn[standard]"
uvicorn rag_eval.serving.app:app --host 0.0.0.0 --port 8000
# then: curl http://localhost:8000/health
```

**Tests:**
```bash
pip install pytest
pytest tests/ -v
```

## Project structure

```
config/settings.yaml     single source of truth for every tunable
data/raw/                untouched downloads
data/manifest/           provenance records (accession numbers, hashes)
data/processed/          parsed/chunked/pooled output + every result table
src/rag_eval/ingest/     corpus acquisition (HotpotQA, EDGAR)
src/rag_eval/pipeline/   Corpus A retrieval + generation pipeline
src/rag_eval/eval/       metrics, ablations, judge calibration (Corpus A)
src/rag_eval/corpus_b/   XBRL extraction, chunking, retrieval, generation (Corpus B)
src/rag_eval/labeling/   local hand-labeling web tool
src/rag_eval/serving/    FastAPI serving layer
tests/                   evaluation-harness regression tests
.github/workflows/       CI (tests + Docker build)
```

See [REPORT.md](REPORT.md) for the full write-up.

