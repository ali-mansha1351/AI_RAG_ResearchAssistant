"""AI RAG Project

A Retrieval-Augmented Generation system for document-based question answering.

This package provides:
- Document processing (PDF, DOCX, TXT)
- Vector storage and retrieval using FAISS
- Language model integration (Groq)
- Research agent for Q&A
- Streamlit web interface
"""

__version__ = "1.0.0"
__author__ = "AI RAG Project"
__description__ = "AI Research Assistant with RAG capabilities"

# Main package imports for convenience
from src.agents import ResearchAgent
from src.database import vector_store
from src.models import get_llm, embedding_manager
from config import settings

__all__ = [
    "ResearchAgent",
    "vector_store", 
    "get_llm",
    "embedding_manager",
    "settings"
]
