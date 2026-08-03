# NVIDIA 2025 Annual Report — RAG Assistant

A production-grade RAG system for question answering over the NVIDIA FY2025 Annual Report. Built with LangGraph, LangChain, Qdrant Cloud, and Groq.

## Architecture

```
User question
    │
    ▼
[rewriter]  — routes the question AND (for RETRIEVE) reformulates it for search
    │
    ├── RETRIEVE ──▶ [responder] — retrieve once (Qdrant hybrid + rerank), then one LLM call to answer
    │                     │
    │                     ▼
    │                [guardrail] — LLM-as-judge grounding check; fallback if unsupported
    │                     │
    ├── CONVERSATIONAL ─▶ [direct_answer] — plain reply for greetings / history, no retrieval
    │                     │
    ├── REFUSE ─────────▶ [refuse] — fixed refusal for off-topic / jailbreak input
    │                     │
    ▼                     ▼
                       Answer
```

**Retrieval pipeline inside [responder]** (item 9: server-side in Qdrant Cloud):
1. Dense prefetch — Vertex `text-embedding-005` (768-dim) vector search
2. Sparse prefetch — BM42 lexical vector search
3. Reciprocal Rank Fusion — merged server-side by Qdrant (top 8 candidates)
4. Cross-encoder reranking (FlashRank ONNX, `ms-marco-MiniLM-L-12-v2`) → top 5 returned to LLM

Ingestion is a separate one-off step (`ci/ingest_qdrant.py`); nothing is built at query time and there is no index round-trip on cold start.

## Stack

| Component | Choice |
|---|---|
| LLM | Groq — `qwen/qwen3.6-27b` (primary), `openai/gpt-oss-20b` (fallback) |
| Embeddings | Vertex AI `text-embedding-005` (via `langchain-google-vertexai`) |
| Vector store | Qdrant Cloud (managed, server-side hybrid search) |
| Lexical search | BM42 sparse vectors (fastembed), fused with dense server-side (RRF) |
| Reranker | FlashRank `ms-marco-MiniLM-L-12-v2` (quantized ONNX; removing torch cut the image 538.9 MB → 379.9 MB) |
| Graph | LangGraph |
| UI | Streamlit |

## Setup

The app is split into two services (item 8): a **FastAPI backend** (`src/api/main.py`, the full RAG pipeline) and a **thin Streamlit UI** (`streamlit_app.py`, which only calls the backend's `/query`).

```bash
# Install dependencies
uv sync

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Terminal 1 — backend. FLASHRANK_CACHE_DIR points the reranker model cache at
# a writable path (the container bakes it at /app/.flashrank_cache; locally you
# must override it, or startup fails trying to write to /app).
FLASHRANK_CACHE_DIR=~/.cache/flashrank uv run uvicorn src.api.main:app --port 8000

# Terminal 2 — thin UI, pointed at the backend
BACKEND_URL=http://localhost:8000 uv run --no-project streamlit run streamlit_app.py
```

Or hit the API directly: `curl -X POST localhost:8000/query -H 'Content-Type: application/json' -d '{"question":"What were NVIDIA 2025 revenues?"}'`. Any client (a Slack bot, a CLI) can consume `/query` the same way.

Retrieval needs a populated Qdrant collection. The backend does **not** build one at startup: ingestion is a separate one-off step (`ci/ingest_qdrant.py`, `--wipe` to drop and recreate), and both local runs and Cloud Run read `QDRANT_URL` / `QDRANT_API_KEY` from the environment. Nothing is indexed at query time and there is no index round-trip on cold start.

## Configuration

All tuneable parameters are in `src/config/config.py`:

| Parameter | Value | Notes |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | 10% of chunk size — prevents boundary meaning loss |
| `LLM_MODEL` | qwen/qwen3.6-27b | Primary Groq model |
| `FALLBACK_MODEL` | openai/gpt-oss-20b | Fallback on primary failure — now used at every LLM call site (rewriter, responder, ground check, direct_answer) |
| `LLM_MAX_RETRIES` | 3 | Retries per model on transient errors |
| `REASONING_FORMAT` | hidden | Suppress reasoning tokens from output (qwen3.6 is a reasoning model) |

## Retrieval Evaluation

25 query/keyword test cases (hit rate / Recall@5, mean context precision), run via Cloud Build.

**`ci/retrieval_eval.py` via the `rag-gcp-eval-manual` trigger.** Queries the live Qdrant collection (the same one Cloud Run serves) and fails the build if hit rate drops below 90%:

```bash
gcloud builds triggers run rag-gcp-eval-manual --branch=rag-gcp --region=southamerica-east1
gcloud builds describe <build-id> --region=southamerica-east1 --format="value(status)"
gcloud builds log <build-id> --region=southamerica-east1
```

Not wired to push — retrieval quality doesn't change on most commits, so this stays manual. Run it after any change to chunk size, embedding model, or reranker settings.

There is also a gitignored local script, `eval/retrieval_eval.py`, from before the Qdrant migration. It still calls `VectorStore.load("faiss_index")`, which no longer exists, so **it does not run** — `ci/retrieval_eval.py` is the only working path. Delete or port it.

**Baseline (Qdrant hybrid, k=8, rerank top_k=5, `text-embedding-005` dense + BM42 sparse):** 100% hit rate, 55% mean context precision (re-measured 2026-08-03). History: 56% on the first Qdrant run (2026-07-18) — a one-chunk difference, i.e. noise; 100%/61% under FAISS+BM25 (held across the item-7 FlashRank swap); 96%/53% under `bge-small-en-v1.5`.

The precision drop from the Qdrant migration looks worse than it is. The 2026-08-03 A/B showed answer quality was **identical** before and after the migration (0.90 groundedness / 4.60 QA quality either side), so those 6 points of context precision were not affecting answers. Hit rate is the number that matters here.

## Answer Quality Evaluation

Retrieval eval only checks whether the *retriever* finds the right chunks — it says nothing about whether the final, user-facing answer is any good. `ci/answer_quality_eval.py`, run via the `rag-gcp-answer-eval-manual` Cloud Build trigger, closes that gap: it runs the real `rewriter -> responder -> guardrail` graph (the same code path `streamlit_app.py` uses) against the first 10 (financial) of the 25 questions (`--limit=10` — the full set can't finish under the TPM cap), then scores the final answers with two Vertex Gen AI eval metrics plus a local key-figure correctness check against golden reference answers:

```bash
gcloud builds triggers run rag-gcp-answer-eval-manual --branch=rag-gcp --region=southamerica-east1
gcloud builds describe <build-id> --region=southamerica-east1 --format="value(status)"
gcloud builds log <build-id> --region=southamerica-east1
```

Three metrics (two Vertex LLM-judge + one local deterministic):

| Metric | Scale | Gate | What it checks |
|---|---|---|---|
| `groundedness` | 0-1 (binary per example, mean = fraction grounded) | ≥ 0.7 | Is the answer supported by retrieved context? Independent check of what `ground_check` already tries to enforce, scored by a separate judge model instead of the app's own Groq call. |
| `question_answering_quality` | 1-5 (rating rubric, 5 = best) | ≥ 3.5 | Is this a good, well-formed answer overall? Broader than groundedness — `ground_check` never checks this. |
| key-figure correctness | 0-1 (fraction of answers containing an accepted PDF-verified figure) | ≥ 0.8 | Reference-based: does the answer state the correct number? Computed locally (no API) against golden answers. Replaced Vertex's `question_answering_correctness`, which was removed from the service (item 6). |

Not wired to push, same reasoning as the retrieval gate. Run manually after prompt, model, or guardrail changes. **Runs `--limit=10` permanently** (the quantitative financial questions) — the full 25-question run can't complete under qwen3.6's 8000 TPM cap; see `known_issues.md`.

> ⚠️ **This gate currently FAILS on `rag-gcp`.** Measured 2026-08-03: groundedness 1.00, QA quality 5.00/5, **key-figure correctness 0.500 against a gate of 0.80**. Five of ten financial answers state no correct figure (EPS, income tax, operating cash flow, data center revenue, gaming revenue). See `known_issues.md`.
>
> The high groundedness and QA-quality scores are misleading, not reassuring. The model answers "I couldn't find that in the report" on the questions it can't handle, and an abstention is trivially grounded and reads as a well-formed answer, so both reference-free judges score it top marks. Only the local key-figure check caught this.

**Cause, isolated by A/B on 2026-08-03:** the 2026-07-18 responder rewrite (ReAct agent → retrieve-once-then-answer). Running the same eval against `d6db045`, the direct parent of that commit, scores **0.900** correctness on the same Qdrant collection with the same script. The agent's ability to re-query with different phrasing was recovering figures buried in the financial tables; retrieve-once gets one shot and abstains when that shot returns a mangled table.

**Historical baselines**, for comparison — note the pre-rewrite control reproduced the FAISS-era numbers exactly, meaning the Qdrant migration did not move answer quality at all:

| Run | Groundedness | QA quality | Key-figure correctness |
|---|---|---|---|
| 2026-07-15, FAISS, pre-rewrite | 0.90 | 4.60/5 | 0.80-0.90 |
| 2026-08-03, Qdrant, pre-rewrite (`d6db045`) | 0.90 | 4.60/5 | 0.900 |
| 2026-08-03, Qdrant, post-rewrite (`17d8bba`) | 1.00 | 5.00/5 | **0.500** |

Requires `GROQ_API_KEY` (Secret Manager, `rag-cloudbuild@` needs `roles/secretmanager.secretAccessor` on it) to generate answers, and the Generative Language API (`generativelanguage.googleapis.com`) enabled on the project for the judge model call — this was the actual blocker the first time this was set up, not an IAM role gap.

## Guardrail / Routing Evaluation

The retrieval and answer gates say nothing about the *input router* (item 4: the classifier folded into the rewriter that sends each question to RETRIEVE, CONVERSATIONAL, or REFUSE). `ci/guardrail_eval.py`, run via the `rag-gcp-guardrail-eval-manual` trigger, measures it: it runs the rewriter node over a labeled input set (on-topic questions, greetings/meta, off-topic + jailbreak strings) and scores predicted vs expected routes.

```bash
gcloud builds triggers run rag-gcp-guardrail-eval-manual --branch=rag-gcp --region=southamerica-east1
```

Cheaper than the other two gates: no Qdrant query at all (the classifier never retrieves, so it runs with a no-op retriever) and no Vertex judge model (scoring is plain TP/FP/FN arithmetic). Only needs `GROQ_API_KEY`. REFUSE is the positive class (catching jailbreaks is the safety-critical job); the gate checks REFUSE precision/recall plus overall accuracy, and a full confusion matrix is printed.

| Metric | Gate | What it checks |
|---|---|---|
| accuracy (all routes) | ≥ 0.85 | Overall fraction of inputs routed correctly |
| REFUSE precision | ≥ 0.90 | When it refuses, is the input actually off-topic/jailbreak? (few false refusals of real questions) |
| REFUSE recall | ≥ 0.80 | Of the inputs that should be refused, how many are caught? |

Not wired to push; run manually after any change to the router prompt or the model. **Last recorded baseline (2026-07-15, qwen/qwen3.6-27b, 36 labeled inputs):** accuracy 1.000, REFUSE precision 1.000, REFUSE recall 1.000 (14/14 retrieve, 8/8 conversational, 14/14 refuse incl. all 8 jailbreak strings). Gates are deliberately kept at 0.90/0.80/0.85 rather than 1.0 — a perfect score on a hand-built set shouldn't turn into a gate that reds the build on one unlucky misroute; the current floor still catches a real regression while tolerating normal LLM variance. Unlike the answer-quality baseline above, this one is still current: routing is decided entirely in the rewriter, which neither the Qdrant migration nor the responder rewrite touched.

## Project Structure

```
src/
  api/main.py       FastAPI backend: /healthz, /query, /feedback, /graph; X-API-Key auth
  app_factory.py    UI-free pipeline bootstrap (build_pipeline) — wires the
                    Qdrant retriever into the graph; no index build or download
  config/           Config class (models, chunking, Qdrant URL/collection/vectors)
  document_ingestion/  PDF loading (incl. gs://) and chunking
  vectorstore/      VectorStore, HybridRetriever (Qdrant), CrossEncoderReranker (FlashRank)
  node/             RAGNodes (rewrite_query + route, generate_answer,
                    ground_check, direct_answer, refuse)
  graph_builder/    LangGraph workflow assembly
  state/            RAGState (Pydantic)
  feedback/         Firestore feedback writes
  logging/          stdout + rotating file logger

ci/
  ingest_qdrant.py        One-off corpus ingestion into Qdrant (--wipe to rebuild)
  retrieval_eval.py       Retrieval eval, exit-code gate, run via Cloud Build
  answer_quality_eval.py  Full-pipeline answer quality gate (Gen AI eval service)
  guardrail_eval.py       Routing/guardrail eval, classifier-only, 36 labeled inputs

data/               Source PDFs (runtime reads the copy in GCS, not this one)
streamlit_app.py    Thin Streamlit client — POSTs to the backend's /query
Dockerfile.api      Full pipeline image (uvicorn)
Dockerfile.ui       Thin UI image (streamlit + requests only)
cloudbuild.yaml                Push-triggered build/deploy of both services
cloudbuild-eval.yaml           Manual retrieval eval gate (rag-gcp-eval-manual)
cloudbuild-answer-eval.yaml    Manual answer-quality gate (rag-gcp-answer-eval-manual)
cloudbuild-guardrail-eval.yaml Manual routing gate (rag-gcp-guardrail-eval-manual)
```

Leftovers not in the runtime path: `Dockerfile` (the pre-item-8 monolith image, still tracked but unused), `faiss_index/` and `eval/` (both gitignored, both dead since the Qdrant migration).
