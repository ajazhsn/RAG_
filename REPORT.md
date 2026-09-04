# RAG + Evaluation Harness: Full Report

A retrieval-augmented generation system built eval-first, with two corpora chosen
specifically to stress-test different failure modes: a benchmark with pre-existing
gold labels (HotpotQA) for clean ablation science, and a real, messy financial
corpus (SEC 10-K filings) with a provenance-verifiable gold set built from
structured XBRL data rather than LLM-generated guesses.

Every number below is from a real run against real data — none are estimated or
illustrative. Raw output backing every table lives under `data/processed/`.

---

## 1. Why two corpora

A benchmark with pre-existing gold labels lets you isolate one variable at a time
(embedding model, retrieval mode, reranker) and trust the resulting number. But it
can't test chunking, since the paragraphs already arrive pre-cut — and it can't
tell you whether your tuned system actually holds up on documents nobody
pre-processed for you.

**Corpus A — HotpotQA (distractor split):** 2,000 multi-hop questions, 19,306
unique paragraphs (deduplicated across questions), gold supporting-fact paragraph
IDs and gold short answers. Used for every ablation that needs a clean, trusted
ground truth: embedding model choice, retrieval mode, reranking.

**Corpus B — SEC 10-K filings (EDGAR):** 99 real, unmodified Inline-XBRL filings,
provenance recorded via CIK + accession number + SHA256 hash. Used for chunking
ablations (impossible on Corpus A) and as a generalization test: does a config
tuned on a clean benchmark still work on real, messy, structurally repetitive
documents?

---

## 2. Corpus A: methodology and results

### 2.1 Naive baseline

Fixed pipeline for a first, untuned number: BGE-small-en-v1.5 embeddings, dense
cosine retrieval, top-5, no reranking, generation via Groq with a citation-required
prompt (cite `[n]` per claim, or explicitly abstain if the passages don't contain
the answer).

| Metric | Value (n=200) |
|---|---|
| Recall@5 | 0.8375 |
| Precision@5 | 0.3407 |
| MRR | 0.9143 |
| nDCG@5 | 0.8181 |
| Answer-contained (generation, lexical floor) | 71.9% |
| Abstention rate | 20% |
| Citation rate | 79.5% |

Precision@5's ceiling is 0.4 here, not 1.0 — HotpotQA questions need exactly 2
supporting facts, so at most 2 of 5 retrieved slots can ever be relevant. Recall
is the metric that matters for this dataset.

"Answer-contained" (does the gold phrase appear verbatim in the generated answer)
is a deliberate lexical floor, not the true accuracy — a correct paraphrase
(e.g. "Otto von Bismarck" for gold "Prussian") scores 0 here despite being right.
This gap is exactly why judge calibration (§2.5) exists.

### 2.2 Retrieval mode ablation: dense vs. BM25 vs. hybrid (n=2,000)

| mode | k | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|
| dense | 5 | 0.8363 | 0.9219 | 0.8161 |
| bm25 | 5 | 0.6260 | 0.7610 | 0.6027 |
| hybrid | 5 | 0.7987 | 0.8820 | 0.7598 |
| dense | 10 | 0.8905 | 0.9227 | 0.8380 |
| hybrid | 10 | **0.9008** | 0.8841 | 0.8008 |
| dense | 20 | 0.9250 | 0.9230 | 0.8488 |
| hybrid | 20 | **0.9323** | 0.8843 | 0.8107 |

**Finding:** dense wins decisively at low k (the practically useful range).
Hybrid only overtakes dense once k≥10 — mixing in BM25's weaker signal actively
hurts top-of-ranking precision, but helps once you're casting a wide net. BM25
alone is the weakest retriever at every k on this dataset, consistent with
HotpotQA's conceptual (not keyword-driven) question style.

### 2.3 Reranker ablation: quality vs. latency (n=300, candidate pool=20, final k=5)

| config | Recall@5 | MRR | nDCG@5 | avg latency/query |
|---|---|---|---|---|
| dense only | 0.8517 | 0.9276 | 0.8328 | 39.5 ms |
| dense + cross-encoder reranker | 0.8933 | 0.9611 | 0.8884 | 7,697.8 ms |

**Finding:** reranking (BAAI/bge-reranker-base) improves every metric — Recall@5
+4.2pts, nDCG@5 +5.6pts — at a **~194x latency cost**. Cross-encoders can't
precompute passage representations the way embeddings can, so every query
re-processes all 20 candidates from scratch. This is a real Pareto tradeoff:
acceptable for offline/batch pipelines, disqualifying for real-time serving.

### 2.4 Embedding model comparison (n=300, k=5)

| model | dim | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| bge-small-en-v1.5 | 384 | 0.8517 | 0.9276 | 0.8328 |
| bge-m3 | 1024 | 0.8367 | 0.9277 | 0.8253 |
| nomic-embed-text-v1.5 | 768 | **0.8600** | 0.9289 | **0.8418** |

**Finding:** nomic-embed-text edges out both alternatives on every metric, but
bge-small (the smallest, cheapest model, CPU-embeddable in ~36 minutes vs.
requiring GPU for the others) is statistically close behind. bge-m3 — the
largest model, ~2.7x bge-small's dimensionality — is actually the **worst**
performer here, a concrete counter-example to "higher MTEB rank/larger dims
implies better task-specific retrieval." bge-small remains the pragmatic default
given the near-tie and dramatically lower compute cost.

### 2.5 Judge calibration: a two-stage story

200 items hand-labeled (faithfulness, correctness) independently, then compared
against an LLM judge (`openai/gpt-oss-120b`, deliberately a different model
family from the generator to avoid self-preference bias).

| Dimension | Raw agreement | Cohen's κ | Interpretation |
|---|---|---|---|
| Correctness | 93.5% | 0.842 | Almost perfect — stable throughout |
| Faithfulness (initial) | 89.0% | 0.314 | Fair — misleadingly high raw agreement |
| Faithfulness (after rubric fix) | 87.5% | 0.349 | Small improvement |
| Faithfulness (after adjudication) | 93.5% | **0.708** | Substantial |

Correctness calibration was strong from the start. Faithfulness was not, and the
process of fixing it is itself the finding:

1. **Rubric gap identified:** the original judge prompt didn't specify how to
   score an *unnecessary* abstention (model had the answer available but
   declined anyway). Clarifying this moved κ only slightly (0.314 → 0.349).
2. **Manual adjudication:** re-reading all 25 faithfulness disagreements against
   source passages revealed 12 genuine hand-labeling errors — factual
   contradictions (a date off by one day, a father/grandfather mix-up) that the
   judge caught on a careful mechanical read but a fast human first-pass missed.
   Correcting these 12 labels (not deferring to the judge — independently
   re-verifying against source text) drove κ to 0.708.

This is why judge calibration is a *process*, not a single number: raw agreement
alone would have overstated trust in the original 0.314-κ judge.

---

## 3. Corpus B: methodology and results

### 3.1 A gold set with no LLM in the loop

Rather than generating candidate Q&A pairs with an LLM and hand-verifying them,
Inline-XBRL filings tag every financial figure with its exact meaning and period
(e.g. `us-gaap:Assets`, context `AsOf2024-12-31`) directly in the document. This
gives a gold answer with **zero generation risk** — extracted, not guessed.

- 99 filings → 54,818 standardized (`us-gaap:*`) facts extracted, with resolved
  period dates and correctly scaled/signed values (verified against synthetic
  fixtures covering scale multipliers, negative signs, nil values, and
  dimensional/segmented contexts before running on real data).
- Filtered to 14 concepts appearing in 70+ of ~99 companies (Assets,
  NetIncomeLoss, StockholdersEquity, EPS, etc.) — excludes fund-specific
  concepts (`InvestmentOwnedAtFairValue` etc.) that only a handful of filers use.
- Filtered out facts with fewer than 4 significant digits (e.g. "10", "42") —
  too generic to be a unique, meaningful retrieval target regardless of matching
  logic sophistication.
- **Final gold set: 243 questions**, from the 20 companies with the best
  concept coverage (all 20 hit 12–14/14 concepts).
- **100% of gold answers confirmed present in visible parsed text** before any
  retrieval was run — a hard precondition checked programmatically, not assumed.

Two real bugs were caught and fixed during qrels construction: plain substring
matching produced false positives (a number embedded inside a larger number,
e.g. "500,000" matching inside "31,500,000") — fixed with boundary-aware regex,
reducing false positives by 92%. A residual issue (very short, generic gold
values matching dozens of unrelated locations) was fixed at the source by the
4-digit filter above, not by further patching the matcher.

### 3.2 Chunking ablation: naive vs. table-aware

The one axis Corpus A structurally could not test, since HotpotQA arrives
pre-chunked. SEC filings mix real financial tables with purely cosmetic
"layout tables" (e.g. the cover page uses `<table>` just to position text) —
this ambiguity is resolved not by up-front classification, but by testing both
chunking philosophies directly:

- **Naive**: flatten all content (text + tables) into one stream per filing,
  fixed 512-word windows with 15% overlap. A table can be split arbitrarily
  wherever a window boundary falls.
- **Table-aware**: text gets the same windowing; every table becomes its own
  atomic chunk, never split, never merged with surrounding prose.

| Pool | Chunks | Atomic table chunks |
|---|---|---|
| Naive | 9,448 | — (tables mixed into text chunks) |
| Table-aware | 22,594 | 13,996 |

### 3.3 Retrieval: naive vs. table-aware, dense vs. BM25 vs. hybrid (n=243)

| pool | mode | k | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|---|
| naive | dense | 5 | 0.1529 | 0.2553 | 0.1631 |
| naive | bm25 | 5 | 0.0843 | 0.1220 | 0.0788 |
| naive | hybrid | 5 | 0.1704 | 0.2362 | 0.1584 |
| table_aware | dense | 5 | **0.1760** | 0.2267 | **0.1672** |
| table_aware | bm25 | 5 | 0.0771 | 0.0935 | 0.0693 |
| table_aware | hybrid | 5 | 0.1651 | 0.2346 | 0.1646 |
| naive | dense | 20 | **0.2639** | 0.2719 | 0.2015 |
| table_aware | dense | 20 | 0.2445 | 0.2374 | 0.1901 |

**Finding 1 — Corpus B is dramatically harder than Corpus A for dense retrieval**
(Recall@5 ~0.15–0.18 vs. Corpus A's 0.84). SEC filings are heavily templated:
nearly every company reports "Total Assets" using near-identical surrounding
language, so dense embeddings — built to capture semantic meaning — struggle to
distinguish *which* company's near-identical statement is meant. This is not a
system bug; it's a real, generalizable property of formulaic documents.

**Finding 2 — BM25 is uniformly the *worst* retriever here too**, contradicting
the initial hypothesis that keyword matching would help on templated language.
When nearly every document shares the same vocabulary, lexical overlap stops
being discriminative — dense embeddings' subtler distinctions, however
imperfect, still beat keyword counting when the keywords themselves carry
little information.

**Finding 3 — table-aware chunking wins at low k, naive wins at high k**
(mirror image of Corpus A's dense/hybrid crossover): intact tables are strong
signals when only a few slots are available, but table-aware's larger pool
(22,594 vs. 9,448 chunks) means more competing near-duplicate tables from other
companies dilute recall once k grows.

### 3.4 Generation and failure taxonomy (n=243)

Correctness here is **objectively computable** — no LLM judge, no human
labeling needed — since gold answers are numbers. A generated answer is scored
correct if any extracted numeric value is within 1% relative tolerance of gold.

| Metric | Value |
|---|---|
| Abstention rate | 46.9% |
| Accuracy on answered | 31.8% (41/129) |

**Failure taxonomy** — every outcome cross-referenced against whether the gold
chunk was actually retrieved, separating retrieval failures from generation
failures:

| Category | Count | % |
|---|---|---|
| Correct | 41 | 16.9% |
| Wrong answer, gold chunk WAS retrieved (generation fault) | 27 | 11.1% |
| Wrong answer, gold chunk NOT retrieved (retrieval fault) | 61 | 25.1% |
| Correct abstention (gold chunk genuinely absent) | 108 | 44.4% |
| Incorrect abstention (gold chunk was retrieved) | 6 | 2.5% |

**Retrieval-attributable failures: 169. Generation-attributable failures: 33.**
An 84%/16% split — once retrieval finds the right chunk, generation is reliable.
The system's bottleneck is squarely retrieval on templated documents, not the
LLM's ability to read a number off a table.

A specific, actionable sub-finding: when retrieval failed, the model chose to
guess a wrong number rather than abstain in 61 of 169 cases (36%) — the
citation-required abstention instruction is measurably less robust when the
model has *some* (irrelevant) context to work with, versus none at all.

**Methodology note:** due to free-tier daily API quota limits, generation used
three models across the 243 items (`openai/gpt-oss-20b`: 132 items,
`qwen/qwen3.6-27b`: 102 items, `openai/gpt-oss-120b`: 9 items), tagged per-item
via a `generator_model` field for full auditability. A per-model breakdown
showed qwen abstained far more often (70.6% vs. 29.5%) but was more accurate
when it did answer (40.0% vs. 31.2%) — a real behavioral difference worth
noting, though not a controlled comparison, since item assignment to each model
was determined by API failure timing, not random assignment.

---

## 4. Serving and CI

- **FastAPI service** (`src/rag_eval/serving/app.py`): `/health` and `/query`
  endpoints, model loaded once at startup. Measured real end-to-end latency on
  a live query: 274.9ms retrieval + 686.5ms generation.
- **Docker**: image builds cleanly in CI; data is deliberately excluded from
  the image and mounted as a volume at runtime, matching real deployment
  separation of code and data.
- **CI** (`.github/workflows/ci.yml`): 25 unit tests covering every metric
  function used across both corpora (Recall@k, nDCG, Cohen's κ, numeric
  extraction) run on every push, plus a Docker build check. Deliberately does
  **not** re-run the full pipeline (embedding, generation) per-commit — that
  would mean spending real API quota and downloading multi-GB models on every
  push, a cost/flakiness tradeoff most production teams avoid. Full pipeline
  re-evaluation is a deliberate, on-demand action.

---

## 5. Key limitations, stated plainly

- Corpus B generation mixed three models due to free-tier quota constraints
  (§3.4) — transparent per-item, not hidden, but not a controlled comparison.
- Chunk size is measured in whitespace-split "words," not real BPE tokens — a
  consistent internal proxy, not claimed to be exact.
- Corpus B's retrieval numbers reflect one embedding model (bge-small) at one
  k-sweep; the embedding-model ablation (§2.4) was only run on Corpus A.
- No bootstrap confidence intervals on ablation deltas — a 4-point Recall@5
  difference is reported as-is, without a formal significance test.

---

## 6. Resume bullets

**Track A — MLE / AI Engineer roles (lead with this project):**

- Built a RAG system with a first-class evaluation harness across two corpora
  (a 19K-paragraph multi-hop benchmark and 99 real SEC filings), running
  retrieval-mode, reranker, and embedding-model ablations with real,
  reproducible numbers — e.g. quantified a cross-encoder reranker's
  quality/latency Pareto tradeoff (+5.6 nDCG@5, ~194x latency cost).
- Designed a provenance-verifiable gold Q&A set from Inline-XBRL financial
  filings (54,818 extracted facts), eliminating LLM-generated-answer risk
  entirely from ground truth construction — includes a documented
  bug-discovery process (92% false-positive reduction in automated qrels
  matching) via boundary-aware string matching.
- Deployed a FastAPI + Docker serving layer with per-request latency
  instrumentation, backed by a CI pipeline (GitHub Actions) running 25
  regression tests on the evaluation harness plus an automated Docker build
  check on every push.

**Track B — Applied Scientist / Data Scientist roles (lead with Publications, then this project):**

- Calibrated an LLM-as-judge against 200 hand-labels via Cohen's κ,
  diagnosing a rubric ambiguity and correcting 12 genuine hand-labeling errors
  through independent source re-verification — raised faithfulness κ from
  0.31 ("fair") to 0.71 ("substantial") while correctness κ remained stable
  at 0.84 throughout.
- Built a failure-attribution taxonomy separating retrieval faults from
  generation faults on a real financial-document RAG system, isolating that
  84% of failures were retrieval-attributable — informing where future
  engineering effort should concentrate rather than guessing.
- Demonstrated that MTEB leaderboard rank does not guarantee task-specific
  performance: a 384-dim embedding model matched or beat a 1024-dim
  alternative on in-domain retrieval, and found BM25 underperforming dense
  retrieval on both a topically-diverse and a heavily templated corpus,
  contradicting an a priori hypothesis about keyword-search advantages on
  formulaic documents.

---

## 7. Where every number lives

```
data/processed/
├── retrieval_scores.json                    naive baseline retrieval metrics
├── generation_scores.json                   naive baseline generation metrics
├── baseline_runs/ablation_retrieval_results.json      dense/bm25/hybrid, Corpus A
├── baseline_runs/ablation_reranker_results.json       reranker tradeoff, Corpus A
├── baseline_runs/ablation_embeddings_results.json     embedding comparison, Corpus A
├── labels/judge_calibration_report.json               kappa + all disagreements
├── sec10k_ablation_retrieval_modes.json               dense/bm25/hybrid, Corpus B
├── sec10k_generation_scores.json                      Corpus B accuracy + per-model
└── sec10k_failure_taxonomy.json                       full per-question categorization
```
