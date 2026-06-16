"""Vector store module for document embedding and retrieval"""
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


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