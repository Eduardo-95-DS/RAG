"""Firestore-backed storage for answer feedback (thumbs up/down)."""
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore

from src.logging.rag_logger import get_logger

log = get_logger()

COLLECTION_NAME = "feedback"

_client: Optional[firestore.Client] = None


def _get_client() -> firestore.Client:
    """Lazily create the Firestore client (reused across calls)."""
    global _client
    if _client is None:
        _client = firestore.Client()
    return _client


def save_feedback(query: str, rewritten_query: str, answer: str, rating: int) -> None:
    """
    Persist a single feedback event to Firestore.

    Args:
        query: The user's original question.
        rewritten_query: The retrieval-optimized rewrite of the question.
        answer: The answer shown to the user.
        rating: 1 for thumbs up, -1 for thumbs down.
    """
    try:
        doc = {
            "query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "rating": rating,
            "timestamp": datetime.now(timezone.utc),
        }
        _get_client().collection(COLLECTION_NAME).add(doc)
        log.info("[FEEDBACK] rating=%d | query='%s'", rating, query[:80])
    except Exception as e:
        # Feedback is a nice-to-have — never let a Firestore hiccup break the UI.
        log.warning("[FEEDBACK] failed to save: %s", str(e))
