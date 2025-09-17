"""Database package for AI RAG Project

Contains vector store implementation and database utilities.
"""

from .vector_store import VectorStore, vector_store

__all__ = [
    "VectorStore",
    "vector_store"
]
