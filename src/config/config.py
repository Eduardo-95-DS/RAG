"""Configuration module for Agentic RAG system"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Model Configuration
    # Primary. Migrated 2026-07-14 from meta-llama/llama-4-scout-17b-16e-instruct,
    # which Groq deprecated (shutdown 2026-07-17). First tried gpt-oss-120b (Groq's
    # other recommended replacement) but it fumbled the ReAct retriever tool schema
    # — generating {"cursor":..,"id":..} instead of {"query":..}, a known Groq
    # gpt-oss tool-call issue — which broke retrieval. qwen3.6-27b is Groq's other
    # recommended replacement and a reliable tool-caller (flagship agentic scores).
    # Caveat: it is Groq *Preview* tier (may change at short notice), not Production;
    # revisit if it's deprecated. It is also a reasoning model, hence
    # reasoning_format="hidden" below so thinking tokens don't leak into the
    # rewriter's query or the ground-check's YES/NO.
    LLM_MODEL = "groq:qwen/qwen3.6-27b"
    # Fallback for transient primary failures (see get_llm_with_fallback). Small,
    # fast, cheap Production model. Only used at the non-tool-calling .invoke()
    # sites (rewriter, ground check), so its own tool-call quirks don't matter here.
    FALLBACK_MODEL = "groq:openai/gpt-oss-20b"
    # Retries per model for transient errors (429/503). Groq's client backs off
    # between attempts; 3 = up to 3 tries before the fallback (if any) engages.
    LLM_MAX_RETRIES = 3

    # Reasoning controls. qwen3.6 defaults to THINKING mode, which was disastrous
    # here: on the terse instruction tasks in this pipeline (query rewrite, YES/NO
    # grounding, tool-driven retrieval) it spent ~926 of ~940 output tokens on
    # reasoning and emitted an EMPTY final answer, producing blank rewrites, empty
    # retrieval, "Could not generate answer.", and multi-minute hangs. Fix per
    # Groq's reasoning docs: reasoning_effort="none" disables reasoning entirely
    # for qwen3.6 (values are none|default, qwen-only). reasoning_format must be
    # "parsed" or "hidden" (not raw) whenever tool calling is on, so "hidden" is
    # kept for the agent path. gpt-oss models don't accept reasoning_format and
    # use a DIFFERENT reasoning_effort scale (low|medium|high), so per-model
    # kwargs are built in _model_kwargs() rather than shared.
    QWEN_REASONING_EFFORT = "none"
    REASONING_FORMAT = "hidden"
    GPTOSS_REASONING_EFFORT = "low"

    @classmethod
    def _model_kwargs(cls, model: str) -> dict:
        """Per-model init kwargs. qwen3.6 and gpt-oss have incompatible reasoning
        params, so branch on the model id rather than passing shared kwargs."""
        kwargs = {"max_retries": cls.LLM_MAX_RETRIES}
        if "qwen" in model:
            kwargs["reasoning_effort"] = cls.QWEN_REASONING_EFFORT
            kwargs["reasoning_format"] = cls.REASONING_FORMAT
        elif "gpt-oss" in model:
            kwargs["reasoning_effort"] = cls.GPTOSS_REASONING_EFFORT
        return kwargs

    # Document Processing
    CHUNK_SIZE = 500
    # Overlap is 10% of chunk size. Enough to prevent meaning loss at boundaries
    # (a sentence split across two chunks remains readable in both) without
    # significantly inflating the total chunk count or retrieval token cost.
    CHUNK_OVERLAP = 50

    # Default sources
    # On rag-gcp, the PDF lives in Cloud Storage instead of the local data/ folder.
    SOURCES = [
        "gs://edu-rag-nvidia-docs/NVIDIA-2025-Annual-Report.pdf"
    ]

    # Qdrant Cloud (item 9): managed vector DB doing hybrid search server-side,
    # replacing local FAISS + BM25 + the GCS index round-trip. Dense vectors are
    # still Vertex text-embedding-005 (768-dim); the lexical channel is BM42
    # sparse vectors (fastembed). QDRANT_URL + QDRANT_API_KEY come from the
    # environment (Secret Manager on Cloud Run, .env locally).
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = "nvidia_annual_report"
    # Vertex text-embedding-005 output dimensionality (dense vector size).
    DENSE_DIM = 768
    # fastembed sparse model for the lexical channel. BM42 is Qdrant's
    # attention-based BM25 successor, tuned for short-text/RAG retrieval —
    # Qdrant's own recommendation over BM25 for this use case.
    SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # Retrieval width. RETRIEVAL_K = fused candidates Qdrant returns after
    # server-side RRF; RERANK_TOP_K = how many FlashRank keeps and the responder
    # puts in the prompt.
    #
    # Raised from 8/5 on 2026-08-03. The 2026-07-18 ReAct removal cut key-figure
    # correctness from 0.900 to 0.500 (see known_issues.md): the agent used to
    # recover figures buried in the financial tables by re-querying, and
    # retrieve-once can only abstain when its single shot returns a mangled
    # table. Widening the one retrieval is the zero-extra-LLM-call fix to try
    # first — if the missing figures sit at ranks 6-10, this recovers them at no
    # TPM cost. RERANK_TOP_K = 8 is the natural ceiling: generate_answer already
    # slices docs[:8] when building the context block.
    RETRIEVAL_K = 16
    RERANK_TOP_K = 8

    @classmethod
    def _require_api_key(cls):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to .env locally or to Streamlit Cloud secrets."
            )

    @classmethod
    def get_llm(cls):
        """Primary LLM with automatic retries on transient errors.

        Returns a plain chat model (not a fallback-wrapped runnable) so it stays
        compatible with create_react_agent, which calls bind_tools() on it — a
        RunnableWithFallbacks does not expose bind_tools. The ReAct agent uses
        this; the plain .invoke() sites use get_llm_with_fallback() instead.
        """
        cls._require_api_key()
        return init_chat_model(cls.LLM_MODEL, **cls._model_kwargs(cls.LLM_MODEL))

    @classmethod
    def get_llm_with_fallback(cls):
        """Primary (with retries) that falls back to a smaller model on failure.

        For plain .invoke() call sites only (rewriter, ground check). Both models
        carry retries; if the primary still fails after its retries, the fallback
        model is tried. Not usable with create_react_agent (see get_llm).
        """
        cls._require_api_key()
        primary = init_chat_model(cls.LLM_MODEL, **cls._model_kwargs(cls.LLM_MODEL))
        fallback = init_chat_model(cls.FALLBACK_MODEL, **cls._model_kwargs(cls.FALLBACK_MODEL))
        return primary.with_fallbacks([fallback])