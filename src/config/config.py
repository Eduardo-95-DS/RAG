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

    # Where the built FAISS index is cached across Cloud Run cold starts.
    # No staleness check: if SOURCES or CHUNK_SIZE/CHUNK_OVERLAP change,
    # delete this prefix in GCS manually to force a rebuild.
    FAISS_INDEX_GCS_PREFIX = "gs://edu-rag-nvidia-docs/faiss_index"

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