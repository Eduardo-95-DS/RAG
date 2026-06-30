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
| LLM | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, CPU) |
| Vector store | FAISS (CPU) |
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
| `LLM_MODEL` | llama-4-scout | Groq model string |

## Retrieval Evaluation

```bash
python eval/retrieval_eval.py
```

Runs 25 query/keyword test cases and reports hit rate (Recall@5) and mean context precision. Use this before and after any change to chunk size, embedding model, or reranker settings.

**Baseline (RRF k=8, rerank top_k=5):** 96% hit rate, 53% mean context precision.

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
  retrieval_eval.py   Offline retrieval quality script

data/               Source PDFs
faiss_index/        Persisted FAISS index (gitignored)
streamlit_app.py    UI entry point
```
