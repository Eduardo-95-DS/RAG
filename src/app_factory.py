"""Shared pipeline bootstrap — builds the RAG graph with no UI dependency.

Item 9 simplified this dramatically: retrieval moved to Qdrant Cloud, so there
is no FAISS index to resolve (local disk / GCS cache / rebuild) at startup
anymore. build_pipeline() just wires the Qdrant-backed retriever into the graph.
The collection is populated out-of-band by ci/ingest_qdrant.py.

Must NOT import streamlit or fastapi — it's the shared core the API sits on.
"""
from src.config.config import Config
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.logging.rag_logger import get_logger

log = get_logger()


def build_pipeline():
    """Build and return (graph_builder, status).

    Retrieval is served by Qdrant Cloud (no index build/download). Raises if the
    Qdrant connection details are missing — callers (FastAPI startup, eval)
    decide how to surface it. `status` is a short string for /healthz.
    """
    llm = Config.get_llm()
    vector_store = VectorStore()
    retriever = vector_store.get_hybrid_retriever(  # validates QDRANT_* env
        k=Config.RETRIEVAL_K, rerank_top_k=Config.RERANK_TOP_K
    )

    graph_builder = GraphBuilder(retriever=retriever, llm=llm)
    graph_builder.build()
    log.info("Pipeline built (retrieval: Qdrant collection '%s')",
             Config.QDRANT_COLLECTION)
    return graph_builder, f"qdrant:{Config.QDRANT_COLLECTION}"
