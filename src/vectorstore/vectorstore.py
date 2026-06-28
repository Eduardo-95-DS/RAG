"""Vector store module for document embedding and retrieval"""
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class VectorStore:
    """Manages vector store operations"""

    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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

    def get_hybrid_retriever(self, k: int = 8) -> "HybridRetriever":
        """Return a HybridRetriever combining FAISS and BM25 with RRF."""
        docs = self.get_all_documents()
        return HybridRetriever(faiss_retriever=self.get_retriever(), documents=docs, k=k)


class HybridRetriever:
    """
    Combines FAISS (semantic) and BM25 (lexical) retrieval using
    Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across both lists.
    """

    RRF_K = 60

    def __init__(self, faiss_retriever, documents: List[Document], k: int = 8):
        self.faiss_retriever = faiss_retriever
        self.k = k
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
        return self._rrf_merge(faiss_docs, bm25_docs)