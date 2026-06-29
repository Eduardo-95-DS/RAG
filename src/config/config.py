"""Configuration module for Agentic RAG system"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Model Configuration
    LLM_MODEL = "groq:meta-llama/llama-4-scout-17b-16e-instruct"

    # Document Processing
    CHUNK_SIZE = 500
    # Overlap is 10% of chunk size. Enough to prevent meaning loss at boundaries
    # (a sentence split across two chunks remains readable in both) without
    # significantly inflating the total chunk count or retrieval token cost.
    CHUNK_OVERLAP = 50

    # Default sources
    SOURCES = [
        "data"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to .env locally or to Streamlit Cloud secrets."
            )
        return init_chat_model(cls.LLM_MODEL)