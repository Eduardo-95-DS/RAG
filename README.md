# NVIDIA 2025 Annual Report — RAG Assistant

A production-grade RAG system for question answering over the NVIDIA FY2025 Annual Report. Built with LangGraph, LangChain, FAISS, and Groq.

## Architecture

```
User question
    │
    ▼
[rewriter]  — resolves conversational references, reformulates for retrieval
    │
    ▼
[responder] — ReAct agent with hybrid retriever tool (FAISS + BM25 + cross-encoder)
    │
    ▼
[guardrail] — LLM-as-judge grounding check; returns fallback if answer is unsupported
    │
    ▼
Answer
```

**Retrieval pipeline inside [responder]:**
1. FAISS semantic search (top 8)
2. BM25 lexical search (top 8)
3. Reciprocal Rank Fusion merge (k=60, top 8 candidates)
4. Cross-encoder reranking → top 5 returned to LLM

## Stack

| Component | Choice |
|---|---|
| LLM | Groq — `qwen/qwen3.6-27b` (primary), `openai/gpt-oss-20b` (fallback) |
| Embeddings | Vertex AI `text-embedding-005` (via `langchain-google-vertexai`) |
| Vector store | FAISS (CPU), persisted to GCS across Cloud Run cold starts |
| Lexical search | BM25 (`rank_bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
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

**Baseline (RRF k=8, rerank top_k=5, `text-embedding-005`):** 100% hit rate, 61% mean context precision. (Superseded the earlier `bge-small-en-v1.5` baseline of 96% / 53% when the branch moved to Vertex embeddings.)

## Answer Quality Evaluation

Retrieval eval only checks whether the *retriever* finds the right chunks — it says nothing about whether the final, user-facing answer is any good. `ci/answer_quality_eval.py`, run via the `rag-gcp-answer-eval-manual` Cloud Build trigger, closes that gap: it runs the real `rewriter -> responder -> guardrail` graph (the same code path `streamlit_app.py` uses) against the same 25 questions, then scores the final answers with the Vertex AI Gen AI evaluation service (LLM-as-judge, reference-free — no golden answers needed):

```bash
gcloud builds triggers run rag-gcp-answer-eval-manual --branch=rag-gcp --region=southamerica-east1
gcloud builds describe <build-id> --region=southamerica-east1 --format="value(status)"
gcloud builds log <build-id> --region=southamerica-east1
```

Two metrics, on two different scales (verified against Vertex's own metric docs — don't assume both are 0-1):

| Metric | Scale | Gate | What it checks |
|---|---|---|---|
| `groundedness` | 0-1 (binary per example, mean = fraction grounded) | ≥ 0.7 | Is the answer supported by retrieved context? Independent check of what `ground_check` already tries to enforce, scored by a separate judge model instead of the app's own Groq call. |
| `question_answering_quality` | 1-5 (rating rubric, 5 = best) | ≥ 3.5 | Is this a good, well-formed answer overall? Broader than groundedness — `ground_check` never checks this. |

Not wired to push, same reasoning as the retrieval gate. Run manually after prompt, model, or guardrail changes.

**Baseline (2026-07-14, qwen/qwen3.6-27b):** groundedness 0.72, question_answering_quality 4.36/5. Both pass the gates (≥ 0.7 / ≥ 3.5), but groundedness dropped from the previous 0.96 (llama-4-scout, 2026-07-13) after the forced model migration — a real faithfulness regression, passing only by a 0.02 margin. Tracked in `known_issues.md`.

Requires `GROQ_API_KEY` (Secret Manager, `rag-cloudbuild@` needs `roles/secretmanager.secretAccessor` on it) to generate answers, and the Generative Language API (`generativelanguage.googleapis.com`) enabled on the project for the judge model call — this was the actual blocker the first time this was set up, not an IAM role gap.

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
