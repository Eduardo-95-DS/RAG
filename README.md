# NVIDIA 2025 Annual Report — RAG Assistant

A production-grade RAG system for question answering over the NVIDIA FY2025 Annual Report. Built with LangGraph, LangChain, FAISS, and Groq.

## Architecture

```
User question
    │
    ▼
[rewriter]  — routes the question AND (for RETRIEVE) reformulates it for search
    │
    ├── RETRIEVE ──▶ [responder] — ReAct agent w/ hybrid retriever (FAISS + BM25 + cross-encoder)
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

**Retrieval pipeline inside [responder]:**
1. FAISS semantic search (top 8)
2. BM25 lexical search (top 8)
3. Reciprocal Rank Fusion merge (k=60, top 8 candidates)
4. Cross-encoder reranking (FlashRank ONNX, `ms-marco-MiniLM-L-12-v2`) → top 5 returned to LLM

## Stack

| Component | Choice |
|---|---|
| LLM | Groq — `qwen/qwen3.6-27b` (primary), `openai/gpt-oss-20b` (fallback) |
| Embeddings | Vertex AI `text-embedding-005` (via `langchain-google-vertexai`) |
| Vector store | FAISS (CPU), persisted to GCS across Cloud Run cold starts |
| Lexical search | BM25 (`rank_bm25`) |
| Reranker | FlashRank `ms-marco-MiniLM-L-12-v2` (quantized ONNX, no torch) |
| Graph | LangGraph |
| UI | Streamlit |

## Setup

```bash
# Install dependencies
uv sync

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app (builds FAISS index on first run)
streamlit run streamlit_app.py
```

The FAISS index is built from PDFs in `data/` on first run and cached to `faiss_index/`. Subsequent runs load from cache.

## Configuration

All tuneable parameters are in `src/config/config.py`:

| Parameter | Value | Notes |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | 10% of chunk size — prevents boundary meaning loss |
| `LLM_MODEL` | qwen/qwen3.6-27b | Primary Groq model |
| `FALLBACK_MODEL` | openai/gpt-oss-20b | Fallback on primary failure (rewriter, ground check) |
| `LLM_MAX_RETRIES` | 3 | Retries per model on transient errors |
| `REASONING_FORMAT` | hidden | Suppress reasoning tokens from output (qwen3.6 is a reasoning model) |

## Retrieval Evaluation

Two ways to run the same 25 query/keyword test cases (hit rate / Recall@5, mean context precision):

**Local, ad hoc** — `eval/retrieval_eval.py`, gitignored, local-only, no gate, just prints a report:

```bash
python eval/retrieval_eval.py
```

Requires a local `faiss_index/` (built by the app, or pulled from GCS yourself).

**Cloud Build, on demand** — `ci/retrieval_eval.py` via the `rag-gcp-eval-manual` trigger. Pulls the live GCS-cached index (the one Cloud Run actually serves), runs the same test cases, and fails the build if hit rate drops below 90%:

```bash
gcloud builds triggers run rag-gcp-eval-manual --branch=rag-gcp --region=southamerica-east1
gcloud builds describe <build-id> --region=southamerica-east1 --format="value(status)"
gcloud builds log <build-id> --region=southamerica-east1
```

Not wired to push — retrieval quality doesn't change on most commits, so this stays manual. Run it after any change to chunk size, embedding model, or reranker settings.

**Baseline (RRF k=8, rerank top_k=5, `text-embedding-005`):** 100% hit rate, 61% mean context precision. Held identically after the item-7 reranker swap to FlashRank `ms-marco-MiniLM-L-12-v2` (2026-07-15) — same numbers as the previous sentence-transformers `ms-marco-MiniLM-L-6-v2`, but without torch. (Both superseded the earlier `bge-small-en-v1.5` baseline of 96% / 53% from before the Vertex-embeddings move.)

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

**Baseline (2026-07-15, qwen/qwen3.6-27b, 10-financial subset):** groundedness 0.90, question_answering_quality 4.60/5, key-figure correctness 0.80-0.90 (9/10 answers carry the right figure; the occasional miss is the guardrail intermittently over-rejecting a financial question to the fallback, not a wrong number). Note: on this financial subset groundedness is 0.90, well above the 0.72 seen on the full 25 — the earlier low figure was concentrated in the open-ended qualitative questions, where the model elaborates beyond the passages.

Requires `GROQ_API_KEY` (Secret Manager, `rag-cloudbuild@` needs `roles/secretmanager.secretAccessor` on it) to generate answers, and the Generative Language API (`generativelanguage.googleapis.com`) enabled on the project for the judge model call — this was the actual blocker the first time this was set up, not an IAM role gap.

## Guardrail / Routing Evaluation

The retrieval and answer gates say nothing about the *input router* (item 4: the classifier folded into the rewriter that sends each question to RETRIEVE, CONVERSATIONAL, or REFUSE). `ci/guardrail_eval.py`, run via the `rag-gcp-guardrail-eval-manual` trigger, measures it: it runs the rewriter node over a labeled input set (on-topic questions, greetings/meta, off-topic + jailbreak strings) and scores predicted vs expected routes.

```bash
gcloud builds triggers run rag-gcp-guardrail-eval-manual --branch=rag-gcp --region=southamerica-east1
```

Cheaper than the other two gates: no FAISS index pull (the classifier never retrieves) and no Vertex judge model (scoring is plain TP/FP/FN arithmetic). Only needs `GROQ_API_KEY`. REFUSE is the positive class (catching jailbreaks is the safety-critical job); the gate checks REFUSE precision/recall plus overall accuracy, and a full confusion matrix is printed.

| Metric | Gate | What it checks |
|---|---|---|
| accuracy (all routes) | ≥ 0.85 | Overall fraction of inputs routed correctly |
| REFUSE precision | ≥ 0.90 | When it refuses, is the input actually off-topic/jailbreak? (few false refusals of real questions) |
| REFUSE recall | ≥ 0.80 | Of the inputs that should be refused, how many are caught? |

Not wired to push; run manually after any change to the router prompt or the model. **Baseline (2026-07-15, qwen/qwen3.6-27b, 36 labeled inputs):** accuracy 1.000, REFUSE precision 1.000, REFUSE recall 1.000 (14/14 retrieve, 8/8 conversational, 14/14 refuse incl. all 8 jailbreak strings). Gates are deliberately kept at 0.90/0.80/0.85 rather than 1.0 — a perfect score on a hand-built set shouldn't turn into a gate that reds the build on one unlucky misroute; the current floor still catches a real regression while tolerating normal LLM variance.

## Project Structure

```
src/
  config/           Config class (model, chunking params)
  document_ingestion/  PDF/URL loading and chunking
  vectorstore/      VectorStore, HybridRetriever, CrossEncoderReranker
  node/             RAGNodes (rewrite_query, generate_answer, ground_check)
  graph_builder/    LangGraph workflow assembly
  state/            RAGState (Pydantic)
  logging/          Rotating file logger

eval/
  retrieval_eval.py   Local-only retrieval quality script (gitignored)

ci/
  retrieval_eval.py       Same retrieval eval, exit-code gate, run via Cloud Build
  answer_quality_eval.py  Full-pipeline answer quality gate (Gen AI eval service)

data/               Source PDFs
faiss_index/        Persisted FAISS index (gitignored)
streamlit_app.py    UI entry point
cloudbuild.yaml             Push-triggered build/deploy (rag-gcp-push-deploy)
cloudbuild-eval.yaml        Manual retrieval eval gate (rag-gcp-eval-manual)
cloudbuild-answer-eval.yaml Manual answer-quality eval gate (rag-gcp-answer-eval-manual)
```
