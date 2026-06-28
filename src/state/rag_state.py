"""RAG state definition for LangGraph"""

from typing import List
from pydantic import BaseModel
# from langchain.schema import Document
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State object for RAG workflow"""

    question: str
    rewritten_query: str = ""
    retrieved_docs: List[Document] = []
    answer: str = ""