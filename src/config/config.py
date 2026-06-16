"""Configuration module for Agentic RAG system"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class Config:
    """Configuration class for RAG system"""
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model Configuration
    LLM_MODEL = "groq:meta-llama/llama-4-scout-17b-16e-instruct"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "data"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return init_chat_model(cls.LLM_MODEL)