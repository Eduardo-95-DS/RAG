"""Shared pipeline bootstrap — builds the RAG graph with no UI dependency.

Extracted from streamlit_app.py's initialize_rag() (item 8) so the FastAPI
backend (src/api/main.py), the eval scripts, and any other caller construct
the exact same pipeline the same way. This module must NOT import streamlit
or fastapi — it's the shared core both sit on top of.
"""
from pathlib import Path

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.logging.rag_logger import get_logger

log = get_logger()

FAISS_INDEX_PATH = "faiss_index"


def build_pipeline(faiss_index_path: str = FAISS_INDEX_PATH):
    """Build and return (graph_builder, index_status).

    Resolves the FAISS index in the same order as before: warm local disk →
    GCS cache → full rebuild from the source PDF (then upload to the GCS cache
    so the next cold start skips it). Returns the compiled GraphBuilder and a
    short status string describing where the index came from
    ("cached", "cached (GCS)", or the chunk count as an int on a fresh build).

    Raises on failure rather than swallowing — callers (FastAPI startup, the
    eval) decide how to surface it. The old Streamlit version caught and
    st.error()'d; that UI concern now lives in the thin client, not here.
    """
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    vector_store = VectorStore()

    if Path(faiss_index_path).exists():
        # Same running container, already warm — local disk still has it.
        vector_store.load(faiss_index_path)
        index_status = "cached"
    elif vector_store.download_from_gcs(Config.FAISS_INDEX_GCS_PREFIX, faiss_index_path):
        # Fresh container, but a prior cold start already built + uploaded it.
        log.info("Loaded FAISS index from GCS cache, skipped rebuild")
        vector_store.load(faiss_index_path)
        index_status = "cached (GCS)"
    else:
        # First cold start ever (or GCS cache cleared): build, then upload.
        documents = doc_processor.process_urls(Config.SOURCES)
        vector_store.create_vectorstore(documents)
        vector_store.save(faiss_index_path)
        index_status = len(documents)
        try:
            vector_store.upload_to_gcs(Config.FAISS_INDEX_GCS_PREFIX, faiss_index_path)
            log.info("Uploaded freshly built FAISS index to GCS cache")
        except Exception as e:
            # Non-fatal: the app works fine off the local index for this
            # container's lifetime; the next cold start just rebuilds again.
            log.warning(f"Failed to upload FAISS index to GCS cache: {e}")

    graph_builder = GraphBuilder(
        retriever=vector_store.get_hybrid_retriever(),
        llm=llm,
    )
    graph_builder.build()
    return graph_builder, index_status
