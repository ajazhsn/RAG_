# rag-eval

RAG system with a first-class evaluation harness — built eval-first, corpus-first,
so every architectural decision produces a measured row in an ablation table
rather than an unverified claim.

## Two corpora, deliberately different roles

- **Corpus A — HotpotQA (distractor split).** Pre-labeled gold supporting facts.
  This is where real IR metrics live: Recall@k, MRR, nDCG against actual qrels,
  not LLM-judged approximations. All ablation sweeps run here.
- **Corpus B — SEC 10-K filings (EDGAR).** Messy real PDFs/HTML, multi-page
  tables, no pre-existing labels. This is where ingestion realism, the
  hand-labeled judge-calibration set, and the generalization check
  (does the Corpus-A-tuned config still work here?) come from.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: fill in GROQ_API_KEY, ANTHROPIC_API_KEY, EDGAR_USER_AGENT
```

## Phase 0 — pull both corpora

```bash
python -m rag_eval.ingest.hotpotqa   # builds data/processed/hotpotqa_{pool,questions,qrels}.jsonl
python -m rag_eval.ingest.edgar      # builds data/manifest/sec_10k_manifest.json + downloads filings
```

`edgar.py` will refuse to run until you set a real `EDGAR_USER_AGENT` in `.env` —
SEC requires a genuine contact string, not a placeholder.

## Project structure

```
config/settings.yaml     single source of truth for every tunable
data/raw/                untouched downloads
data/manifest/           provenance records (accession numbers, hashes)
data/processed/          parsed/chunked/pooled output
src/rag_eval/ingest/     corpus acquisition scripts
src/rag_eval/utils/      shared helpers (logging, etc.)
notebooks/                scratch exploration only — nothing load-bearing lives here
tests/
```

See project plan (shared separately) for the full phase breakdown and ablation axes.
