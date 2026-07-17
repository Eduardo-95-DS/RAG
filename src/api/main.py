"""FastAPI backend for the NVIDIA 2025 Annual Report RAG assistant (item 8).

The pipeline (rewriter -> responder -> guardrail, hybrid retrieval, guardrail)
now lives behind a REST API instead of inside the Streamlit process. The thin
Streamlit client, the eval, or any other consumer (a Slack bot, a CLI) calls
POST /query. The pipeline is built ONCE at startup via app_factory.build_pipeline.

Auth: both Cloud Run services are public, but /query and /feedback require a
shared secret header (X-API-Key) checked against the API_KEY env var, so a
random caller can't burn the Groq quota through the open endpoint. If API_KEY
is unset (local dev), the check is skipped.
"""
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from src.app_factory import build_pipeline
from src.node.reactnode import RAGNodes
from src.feedback.feedback_store import save_feedback
from src.logging.rag_logger import get_logger

log = get_logger()

API_KEY = os.getenv("API_KEY")  # shared header key; if unset, auth is skipped

# Built once at startup (see lifespan below).
_graph = None
_index_status = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once when the container starts (modern replacement
    for the deprecated @app.on_event('startup'))."""
    global _graph, _index_status
    log.info("[API] building pipeline at startup...")
    _graph, _index_status = build_pipeline()
    log.info("[API] pipeline ready (index=%s)", _index_status)
    yield


app = FastAPI(
    title="NVIDIA 2025 Annual Report RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


# --- request / response models --------------------------------------------

class Turn(BaseModel):
    q: str
    a: str


class QueryRequest(BaseModel):
    question: str
    history: List[Turn] = []


class Source(BaseModel):
    text: str
    source: str = ""
    page: str = ""


class QueryResponse(BaseModel):
    answer: str
    rewritten_query: str = ""
    route: str = "retrieve"
    grounded: bool = True
    sources: List[Source] = []
    elapsed_s: float = 0.0


class FeedbackRequest(BaseModel):
    query: str
    rewritten_query: str = ""
    answer: str
    rating: int  # 1 (up) or -1 (down)


# --- auth helper -----------------------------------------------------------

def _check_api_key(x_api_key: Optional[str]):
    """Reject the request if API_KEY is configured and the header doesn't match.
    No-op when API_KEY is unset (local dev)."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --- endpoints -------------------------------------------------------------

@app.get("/healthz")
def healthz():
    """Liveness + readiness. ready=True only once the pipeline is built."""
    return {"status": "ok", "ready": _graph is not None, "index": str(_index_status)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, x_api_key: Optional[str] = Header(default=None)):
    """Run the full pipeline and return the answer + transparency payload."""
    _check_api_key(x_api_key)
    if _graph is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    if len(req.question) > 500:
        raise HTTPException(status_code=422, detail="Question too long (max 500 chars)")

    history = [{"q": t.q, "a": t.a} for t in req.history[-3:]]
    t0 = time.time()
    result = _graph.run(req.question, history=history)
    elapsed = time.time() - t0

    answer = result.get("answer", "")
    retrieved = result.get("retrieved_docs") or []
    sources = [
        Source(
            text=d.page_content,
            source=(d.metadata or {}).get("source", ""),
            page=str((d.metadata or {}).get("page", "")),
        )
        for d in retrieved
    ]
    return QueryResponse(
        answer=answer,
        rewritten_query=result.get("rewritten_query", ""),
        route=result.get("route", "retrieve"),
        grounded=answer != RAGNodes.FALLBACK_ANSWER,
        sources=sources,
        elapsed_s=round(elapsed, 2),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest, x_api_key: Optional[str] = Header(default=None)):
    """Persist a thumbs up/down. Wraps save_feedback (never breaks the caller)."""
    _check_api_key(x_api_key)
    save_feedback(
        query=req.query,
        rewritten_query=req.rewritten_query,
        answer=req.answer,
        rating=1 if req.rating == 1 else -1,
    )
    return {"status": "ok"}


@app.get("/graph")
def graph():
    """Return the LangGraph diagram as a PNG (best-effort; optional)."""
    if _graph is None or _graph.graph is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    try:
        png = _graph.graph.get_graph().draw_mermaid_png()
        return Response(content=png, media_type="image/png")
    except Exception as e:
        # draw_mermaid_png needs optional deps / network; don't hard-fail.
        raise HTTPException(status_code=501, detail=f"Graph rendering unavailable: {e}")
