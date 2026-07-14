"""RAG state definition for LangGraph"""

from typing import Any, Dict, List
from pydantic import BaseModel
from langchain_core.documents import Document


class RAGState(BaseModel):
    """State object for RAG workflow"""

    question: str
    rewritten_query: str = ""
    retrieved_docs: List[Document] = []
    answer: str = ""
    # Routing decision from the rewriter node: "retrieve" (technical question,
    # run the full pipeline), "conversational" (greeting/history, answer directly),
    # or "refuse" (off-topic/jailbreak, polite refusal). Defaults to retrieve so
    # any parse ambiguity degrades to the original behavior.
    route: str = "retrieve"
    # Each entry is {"q": str, "a": str} — the last N turns from the session.
    # Used by the rewriter to resolve conversational references like "that" or "them".
    conversation_history: List[Dict[str, Any]] = []