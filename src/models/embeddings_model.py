"""Embedding models for the research assistant."""

from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding models for the research assistant."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or settings.embedding_model
        self._embeddings = None
        self._sentence_transformer = None
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get LangChain compatible embeddings."""
        if self._embeddings is None:
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Loaded HuggingFace embeddings: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {e}")
                # Fallback to a basic model
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        
        return self._embeddings
    
    def get_sentence_transformer(self) -> SentenceTransformer:
        """Get direct SentenceTransformer model for more control."""
        if self._sentence_transformer is None:
            try:
                self._sentence_transformer = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                # Fallback
                self._sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        
        return self._sentence_transformer
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        model = self.get_sentence_transformer()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings."""
        model = self.get_sentence_transformer()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_documents_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """Process embeddings in batches for better performance.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to log progress
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        all_embeddings = []
        model = self.get_sentence_transformer()
        
        try:
            import streamlit as st
            # Check if we're in a Streamlit context
            if hasattr(st, 'session_state'):
                progress_placeholder = st.empty()
            else:
                progress_placeholder = None
        except:
            progress_placeholder = None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Process batch
                batch_embeddings = model.encode(
                    batch, 
                    normalize_embeddings=True,
                    show_progress_bar=False  # We'll handle progress ourselves
                )
                all_embeddings.extend(batch_embeddings.tolist())
                
                if show_progress:
                    processed = min(i + batch_size, len(texts))
                    progress_pct = processed / len(texts)
                    
                    # Update Streamlit progress if available
                    if progress_placeholder:
                        progress_placeholder.text(f"🧠 Generating embeddings: {processed}/{len(texts)} ({progress_pct*100:.1f}%)")
                    
                    logger.info(f"Processed embeddings: {processed}/{len(texts)} ({progress_pct*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed batch to maintain index alignment
                empty_dim = len(all_embeddings[0]) if all_embeddings else 384  # Default dimension
                for _ in batch:
                    all_embeddings.append([0.0] * empty_dim)
        
        # Clear progress placeholder
        if progress_placeholder:
            progress_placeholder.empty()
        
        logger.info(f"Completed embedding generation for {len(texts)} documents")
        return all_embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        model = self.get_sentence_transformer()
        embeddings = model.encode([text1, text2], normalize_embeddings=True)
        similarity = model.similarity(embeddings[0:1], embeddings[1:2])
        return float(similarity[0][0])


# Global embedding manager instance
embedding_manager = EmbeddingManager()