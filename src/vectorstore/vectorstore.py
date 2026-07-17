"""Retrieval module (item 9): Qdrant Cloud hybrid search + FlashRank rerank.

Replaces the previous local FAISS + BM25 + GCS-index implementation. Retrieval
now runs server-side in Qdrant: a dense (Vertex text-embedding-005) prefetch and
a sparse (BM42) prefetch, fused with RRF by Qdrant, then reranked locally by the
FlashRank cross-encoder. Ingestion is a separate one-off step (ci/ingest_qdrant.py);
nothing is built at query time and there is no index round-trip on cold start.

The public surface is unchanged: `VectorStore().get_hybrid_retriever()` returns a
`HybridRetriever` whose `.invoke(query)` returns a list of `Document`, so
`RAGNodes` / `GraphBuilder` are untouched.
"""
import os
from typing import List

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

from src.config.config import Config


class VectorStore:
    """Provides a hybrid retriever backed by a Qdrant Cloud collection.

    Much thinner than the old FAISS version: no index build/save/load and no GCS
    round-trip. The collection is populated out-of-band by ci/ingest_qdrant.py.
    Embeddings are still Vertex text-embedding-005 via ADC (IAM), unchanged — see
    conventions.md for why VertexAIEmbeddings stays despite its deprecation.
    """

    def __init__(self):
        self._dense = VertexAIEmbeddings(model_name="text-embedding-005")

    def get_hybrid_retriever(self, k: int = 8, rerank_top_k: int = 5) -> "HybridRetriever":
        """Return a HybridRetriever over the Qdrant collection.

        k = number of candidates Qdrant returns after server-side RRF fusion;
        rerank_top_k = how many the local FlashRank reranker keeps.
        """
        if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set (Secret Manager on "
                "Cloud Run, .env locally). Qdrant holds the retrieval index now."
            )
        client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        return HybridRetriever(
            client=client,
            dense_embedder=self._dense,
            k=k,
            rerank_top_k=rerank_top_k,
        )


class CrossEncoderReranker:
    """
    Reranks a candidate set of documents using a cross-encoder model.

    A cross-encoder takes (query, document) pairs and produces a relevance
    score for each pair jointly — unlike a bi-encoder, which encodes query and
    document independently. Joint encoding is slower but significantly more
    accurate for ranking, which is why cross-encoders are used as a second-stage
    reranker rather than a first-stage retriever.

    Runtime: FlashRank (item 7) — a quantized ONNX cross-encoder (onnxruntime,
    no torch). Model ms-marco-MiniLM-L-12-v2 (FlashRank ships no L-6-v2).

    CRITICAL: model_name must be passed explicitly. Ranker() with no args
    defaults to ms-marco-TinyBERT-L-2-v2, a much weaker model.
    """

    MODEL = "ms-marco-MiniLM-L-12-v2"
    # Cache dir order: arg > FLASHRANK_CACHE_DIR env > in-image default. The
    # Docker image bakes the model at the default path; local/CI runs override.
    DEFAULT_CACHE_DIR = "/app/.flashrank_cache"

    def __init__(self, top_k: int = 5, cache_dir: str | None = None):
        from flashrank import Ranker
        cache_dir = cache_dir or os.getenv("FLASHRANK_CACHE_DIR", self.DEFAULT_CACHE_DIR)
        self._ranker = Ranker(model_name=self.MODEL, cache_dir=cache_dir)
        self.top_k = top_k

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Score every (query, doc) pair and return the top_k highest-scoring docs."""
        from flashrank import RerankRequest
        if not docs:
            return docs
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        ranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[p["id"]] for p in ranked[: self.top_k]]


class HybridRetriever:
    """
    Hybrid retrieval via Qdrant's Query API: a dense prefetch (Vertex
    text-embedding-005) and a sparse prefetch (BM42), fused server-side with
    Reciprocal Rank Fusion, then reranked locally by the FlashRank cross-encoder.

    Keeps the `.invoke(query) -> List[Document]` interface of the old FAISS+BM25
    retriever, so RAGNodes / GraphBuilder don't change.
    """

    def __init__(self, client: QdrantClient, dense_embedder, k: int = 8,
                 rerank_top_k: int = 5):
        self.client = client
        self.dense = dense_embedder
        self.k = k  # candidates after server-side fusion, before local rerank
        self._reranker = CrossEncoderReranker(top_k=rerank_top_k)
        # BM42 sparse embedder for the query's lexical vector. Loaded lazily-ish
        # here (once per retriever) — same model the ingestion used.
        self._sparse = SparseTextEmbedding(model_name=Config.SPARSE_MODEL)

    def _sparse_query(self, query: str) -> models.SparseVector:
        sv = next(iter(self._sparse.embed([query])))
        return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())

    def invoke(self, query: str) -> List[Document]:
        dense_vec = self.dense.embed_query(query)
        sparse_vec = self._sparse_query(query)

        # Server-side hybrid: prefetch top candidates from each channel, then
        # fuse with RRF. Ask for `k` fused candidates to hand to the reranker.
        result = self.client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=self.k),
                models.Prefetch(query=sparse_vec, using="sparse", limit=self.k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.k,
            with_payload=True,
        )

        docs: List[Document] = []
        for point in result.points:
            payload = point.payload or {}
            docs.append(Document(
                page_content=payload.get("text", ""),
                metadata={"source": payload.get("source", ""),
                          "page": payload.get("page", "")},
            ))
        return self._reranker.rerank(query, docs)
