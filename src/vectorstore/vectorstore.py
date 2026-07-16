"""Vector store module for document embedding and retrieval"""
import os
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from google.cloud import storage
from google.cloud.exceptions import NotFound


class VectorStore:
    """Manages vector store operations"""

    # VertexAIEmbeddings is deprecated in langchain-google-vertexai in favor of
    # GoogleGenerativeAIEmbeddings, but that class only documents the preview
    # gemini-embedding-2-preview model and requires a separate GOOGLE_API_KEY
    # (Gemini Developer API auth) instead of the IAM/service-account auth used
    # by every other GCP service in this project (Storage, Secret Manager,
    # Firestore, Cloud Run). VertexAIEmbeddings still works, still supports
    # the stable text-embedding-005 model, and uses the same ADC auth as
    # everything else, so it stays despite the deprecation warning.
    def __init__(self):
        self.embedding = VertexAIEmbeddings(model_name="text-embedding-005")
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def save(self, path: str = "faiss_index"):
        """Save vector store to disk"""
        if self.vectorstore is None:
            raise ValueError("Nothing to save. Call create_vectorstore first.")
        self.vectorstore.save_local(path)

    def load(self, path: str = "faiss_index"):
        """Load vector store from disk"""
        self.vectorstore = FAISS.load_local(
            path, self.embedding, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()

    # -- GCS persistence -----------------------------------------------
    #
    # Cold start on Cloud Run has no local disk to reuse (each container
    # instance starts empty), so "save/load from disk" alone doesn't help
    # across cold starts the way it does for a long-lived local dev server.
    # These two methods let a built index survive across cold starts by
    # round-tripping the two files FAISS.save_local() writes (index.faiss,
    # index.pkl) through a GCS prefix instead.
    #
    # No staleness check: this assumes the corpus and chunking config never
    # change without a manual index rebuild. If SOURCES or Config.CHUNK_SIZE/
    # CHUNK_OVERLAP ever change, delete the GCS prefix manually to force a
    # rebuild — otherwise the next cold start will keep loading the stale
    # cached index instead of re-embedding.
    _INDEX_FILES = ("index.faiss", "index.pkl")

    def download_from_gcs(self, gcs_prefix: str, local_path: str = "faiss_index") -> bool:
        """
        Download a previously-built FAISS index from a GCS prefix, if present.

        Args:
            gcs_prefix: URI in the form gs://bucket-name/some/prefix
                        (both index.faiss and index.pkl are expected there)
            local_path: local directory to download into

        Returns:
            True if both index files were found and downloaded, False if
            either is missing (caller should fall back to building fresh).
        """
        if not gcs_prefix.startswith("gs://"):
            raise ValueError(f"Not a GCS URI: {gcs_prefix}")

        bucket_name, _, prefix = gcs_prefix[len("gs://"):].partition("/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        Path(local_path).mkdir(parents=True, exist_ok=True)
        for filename in self._INDEX_FILES:
            blob = bucket.blob(f"{prefix.rstrip('/')}/{filename}")
            try:
                blob.download_to_filename(str(Path(local_path) / filename))
            except NotFound:
                return False
        return True

    def upload_to_gcs(self, gcs_prefix: str, local_path: str = "faiss_index") -> None:
        """
        Upload a locally-built FAISS index (index.faiss, index.pkl) to a GCS
        prefix, so the next cold start can download it instead of rebuilding.

        Failures are not swallowed here — if this fails, the caller decides
        whether that's fatal (see initialize_rag() in streamlit_app.py, which
        logs and continues since the app still works from the local index).
        """
        if not gcs_prefix.startswith("gs://"):
            raise ValueError(f"Not a GCS URI: {gcs_prefix}")

        bucket_name, _, prefix = gcs_prefix[len("gs://"):].partition("/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for filename in self._INDEX_FILES:
            blob = bucket.blob(f"{prefix.rstrip('/')}/{filename}")
            blob.upload_from_filename(str(Path(local_path) / filename))

    def get_retriever(self):
        """Get the retriever instance"""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.vectorstore.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """Return all documents stored in the FAISS index."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        docstore = self.vectorstore.docstore
        return [docstore.search(doc_id) for doc_id in self.vectorstore.index_to_docstore_id.values()]

    def get_hybrid_retriever(self, k: int = 8, rerank_top_k: int = 5) -> "HybridRetriever":
        """Return a HybridRetriever combining FAISS and BM25 with RRF, then reranked."""
        docs = self.get_all_documents()
        return HybridRetriever(
            faiss_retriever=self.get_retriever(),
            documents=docs,
            k=k,
            rerank_top_k=rerank_top_k,
        )


class CrossEncoderReranker:
    """
    Reranks a candidate set of documents using a cross-encoder model.

    A cross-encoder takes (query, document) pairs and produces a relevance
    score for each pair jointly — unlike a bi-encoder (e.g. BGE), which
    encodes query and document independently. Joint encoding is slower but
    significantly more accurate for ranking, which is why cross-encoders
    are used as a second-stage reranker rather than a first-stage retriever.

    Runtime: FlashRank (item 7, 2026-07-15) — a quantized ONNX cross-encoder,
    replacing the previous sentence-transformers CrossEncoder. This removes
    torch and sentence-transformers from the image entirely (onnxruntime is
    far smaller), shrinking the container and cold start.

    Model: ms-marco-MiniLM-L-12-v2
      - The MiniLM cross-encoder FlashRank ships (its model_file_map has no
        L-6-v2, the sentence-transformers model this branch used before — so
        this is a deliberate UPGRADE to the stronger 12-layer variant, not a
        like-for-like swap; retrieval eval was re-measured after the change).
      - Quantized ONNX (~34 MB), CPU, no API call, no torch.

    CRITICAL: model_name must be passed explicitly. Ranker() with no args
    defaults to ms-marco-TinyBERT-L-2-v2, a much weaker model.
    """

    MODEL = "ms-marco-MiniLM-L-12-v2"

    # Cache dir resolution order: explicit arg > FLASHRANK_CACHE_DIR env >
    # a stable in-image default (/app/.flashrank_cache). The Docker image bakes
    # the model to the default path at build time so cold starts don't
    # re-download it. Local runs / CI set FLASHRANK_CACHE_DIR (or pass cache_dir)
    # to a writable path, since /app isn't writable off-container.
    DEFAULT_CACHE_DIR = "/app/.flashrank_cache"

    def __init__(self, top_k: int = 5, cache_dir: str | None = None):
        cache_dir = cache_dir or os.getenv("FLASHRANK_CACHE_DIR", self.DEFAULT_CACHE_DIR)
        # Explicit model_name — see CRITICAL note above.
        self._ranker = Ranker(model_name=self.MODEL, cache_dir=cache_dir)
        self.top_k = top_k

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Score every (query, doc) pair and return the top_k highest-scoring docs."""
        if not docs:
            return docs
        # Carry the original index in each passage so we can map FlashRank's
        # score-sorted output back to the original Document objects (FlashRank
        # returns dicts, not Documents).
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(docs)
        ]
        ranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[p["id"]] for p in ranked[: self.top_k]]


class HybridRetriever:
    """
    Combines FAISS (semantic) and BM25 (lexical) retrieval using
    Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across both lists.
    After RRF merging, a cross-encoder reranker narrows the candidate set
    from k down to rerank_top_k before returning.
    """

    RRF_K = 60

    def __init__(
        self,
        faiss_retriever,
        documents: List[Document],
        k: int = 8,
        rerank_top_k: int = 5,
    ):
        self.faiss_retriever = faiss_retriever
        self.k = k  # number of RRF candidates to generate
        self._reranker = CrossEncoderReranker(top_k=rerank_top_k)
        self._docs = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str) -> List[Document]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._docs[i] for i in ranked[: self.k]]

    def _rrf_merge(
        self, faiss_docs: List[Document], bm25_docs: List[Document]
    ) -> List[Document]:
        scores: dict[str, float] = {}
        id_to_doc: dict[str, Document] = {}

        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            id_to_doc[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            id_to_doc[key] = doc

        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [id_to_doc[k] for k in ranked[: self.k]]

    def invoke(self, query: str) -> List[Document]:
        faiss_docs = self.faiss_retriever.invoke(query)
        bm25_docs = self._bm25_search(query)
        candidates = self._rrf_merge(faiss_docs, bm25_docs)  # up to k=8
        return self._reranker.rerank(query, candidates)       # narrowed to rerank_top_k=5