"""Configuration module for Agentic RAG system"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Model Configuration
    #
    # Primary migrated 2026-08-03 from `groq:qwen/qwen3.6-27b` to Anthropic. The
    # move is about MEASURABILITY, not model quality: Groq's free tier caps
    # qwen3.6-27b at 8000 TPM, which (a) forced the answer eval to run
    # `--limit=10`, pinning per-question noise at 0.100 — wider than the effects
    # being measured — and (b) is suspected of silently routing calls to the
    # FALLBACK model once the primary exhausts its retries, which would make
    # every sequential A/B on that tier invalid. See known_issues.md items 1, 2, 5.
    # Groq's Dev tier (10x limits) was unavailable, so the provider changed.
    #
    # Claude Haiku 4.5 on Tier 1: ~50 RPM, ~50,000 ITPM. The full 25-question eval
    # is ~70k input tokens, so it fits WITHOUT pacing — per-question noise drops
    # 0.100 -> 0.040. RPM is the tighter constraint (25 questions x 3 calls = 75
    # requests), so keep the inter-question sleep.
    #
    # Cost: ~$0.0037/query ($1.00/M in, $5.00/M out) vs ~$0.0022 on qwen. More
    # expensive per token, but a full eval run is ~$0.09 and existing credits
    # cover ~50 of them. Revisit if this ever takes real traffic.
    #
    # NOTE: every eval baseline recorded before this date was measured on
    # qwen3.6-27b. They are historical. Do not compare across this line — how
    # readily a model extracts a figure from mangled table text vs. declining to
    # answer is model-specific, and abstention is exactly our failure mode.
    LLM_MODEL = "anthropic:claude-haiku-4-5-20251001"
    # Fallback for transient primary failures (see get_llm_with_fallback).
    # Deliberately kept on a DIFFERENT provider: a fallback that shares the
    # primary's rate limiter isn't a fallback. gpt-oss-20b on Groq's free tier
    # costs nothing and covers an Anthropic outage or 429 burst.
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

    # Deterministic decoding. Added 2026-08-03; before this, every call ran at
    # Groq's default sampling temperature, which was never a deliberate choice.
    #
    # Nothing in this pipeline is a creative task: the rewriter emits a fixed
    # ROUTE line plus a search query, the responder extracts figures verbatim
    # from retrieved passages, and the ground check answers one word. Sampling
    # buys nothing and costs reproducibility — two runs of the SAME commit
    # scored 0.700 and 0.800 on the 10-question answer eval (one question of
    # swing, since n=10 makes each question worth 0.100), with individual
    # questions flipping pass/fail between runs on identical code. That makes
    # the gate a coin flip at the threshold and makes any A/B smaller than
    # ~0.200 unmeasurable. It also means the live app gives different answers
    # to the same question on different days, which is its own problem for a
    # financial-document assistant.
    #
    # Note temperature=0 reduces variance sharply but does not guarantee bitwise
    # determinism: provider-side batching and float non-associativity can still
    # shift a token. Expect stable-not-identical. (Under qwen3.6-27b it demonstrably
    # did NOT settle: two agreeing runs were followed by a disagreeing one.)
    #
    # Provider note: Anthropic models accept `temperature` directly. OpenAI's
    # GPT-5 line does NOT — it 400s unless reasoning effort is "none" first — which
    # is one reason the swap went to Anthropic rather than gpt-5.6-luna.
    TEMPERATURE = 0.0

    @classmethod
    def _model_kwargs(cls, model: str) -> dict:
        """Per-model init kwargs. Reasoning params are NOT portable across models —
        qwen3.6, gpt-oss and Anthropic each want something different, and passing
        the wrong one is a hard 400, so branch on the model id.

        Anthropic (current primary) needs no reasoning kwargs at all: it takes
        `temperature` directly, which is the base case. The qwen/gpt-oss branches
        are retained because gpt-oss-20b is still the cross-provider fallback,
        and because reverting the primary to Groq should not require re-deriving
        any of this (it cost a full debugging session the first time — see
        tech_stack.md)."""
        kwargs = {"max_retries": cls.LLM_MAX_RETRIES, "temperature": cls.TEMPERATURE}
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
    # DO NOT raise this to chase the failing table questions — that experiment was
    # run on 2026-08-05 and answered. ci/inspect_fusion.py put the
    # operating-cash-flow chunk at fused rank 35, so at k=16 it never reached the
    # reranker. Raising k to 50 DID place it in the candidate pool, and FlashRank
    # still left it out of the top 8. So the bottleneck is the cross-encoder, not
    # prefetch depth, and k=50 only bought ~3x the local reranking work per query
    # (a real cost against the ~7.4s median) for no recovered answer. Reverted.
    #
    # The remaining lever for those questions is the reranker itself — see
    # known_issues.md item 4.
    RETRIEVAL_K = 16
    # generate_answer slices docs[:RERANK_TOP_K] into the prompt, so raising THIS
    # costs prompt tokens on every query — unlike RETRIEVAL_K, which costs none.
    # Raised 5 -> 8 on 2026-08-03 (worth ~2 questions of correctness); 8 is the
    # practical ceiling without also touching generate_answer.
    RERANK_TOP_K = 8

    # Env var holding the key for each provider prefix used in a model string.
    PROVIDER_KEY_ENV = {
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    @classmethod
    def _require_api_key(cls, *models: str):
        """Fail fast, per provider, naming the exact variable that's missing.

        Since the primary and the fallback now live on DIFFERENT providers, a
        single GROQ_API_KEY check would let the app start with no way to reach
        its primary and only discover it on the first user question.
        """
        missing = []
        for model in models:
            provider = model.split(":", 1)[0]
            env_var = cls.PROVIDER_KEY_ENV.get(provider)
            if env_var and not os.getenv(env_var) and env_var not in missing:
                missing.append(env_var)
        if missing:
            raise ValueError(
                f"{', '.join(missing)} not set. Add to .env locally, or to "
                "Secret Manager + the cloudbuild deploy step for Cloud Run."
            )

    @classmethod
    def get_llm(cls):
        """Primary LLM with automatic retries on transient errors.

        Returns a plain, unwrapped chat model. Historically this existed because
        `create_react_agent` calls `bind_tools()`, which a RunnableWithFallbacks
        doesn't expose. The agent was removed 2026-07-18, so every call site now
        uses get_llm_with_fallback() instead — this is kept only because
        GraphBuilder still takes an `llm` argument.
        """
        cls._require_api_key(cls.LLM_MODEL)
        return init_chat_model(cls.LLM_MODEL, **cls._model_kwargs(cls.LLM_MODEL))

    @classmethod
    def get_llm_with_fallback(cls):
        """Primary (with retries) falling back to a second model on failure.

        Used at every .invoke() site: rewriter, responder, ground check,
        direct_answer. Both models carry retries; if the primary still fails
        after its own, the fallback is tried.

        The fallback is on a different provider on purpose (see FALLBACK_MODEL).
        Note it engages SILENTLY — LangChain surfaces no signal that it fired,
        which is why RAGNodes logs the answering model from response metadata.
        A silent fallback is the leading suspect for the 2026-08-03 eval
        instability; don't remove that logging.
        """
        cls._require_api_key(cls.LLM_MODEL, cls.FALLBACK_MODEL)
        primary = init_chat_model(cls.LLM_MODEL, **cls._model_kwargs(cls.LLM_MODEL))
        fallback = init_chat_model(cls.FALLBACK_MODEL, **cls._model_kwargs(cls.FALLBACK_MODEL))
        return primary.with_fallbacks([fallback])