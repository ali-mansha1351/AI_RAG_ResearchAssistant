"""Vector store implementation using FAISS."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss

from src.models.embeddings_model import embedding_manager
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using FAISS."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = persist_directory or str(settings.vector_store_path)
        self.embeddings = embedding_manager.get_embeddings()
        self._vectorstore = None
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create the persistence directory if it doesn't exist."""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def vectorstore(self) -> FAISS:
        """Get or create the FAISS vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create_vectorstore()
        return self._vectorstore
    
    def _load_or_create_vectorstore(self) -> FAISS:
        """Load existing vectorstore or create a new one."""
        index_path = Path(self.persist_directory) / "index.faiss"
        docstore_path = Path(self.persist_directory) / "index.pkl"
        
        logger.info(f"Checking for existing vector store at: {self.persist_directory}")
        logger.info(f"Index file exists: {index_path.exists()}")
        logger.info(f"Docstore file exists: {docstore_path.exists()}")

        if index_path.exists() and docstore_path.exists():
            try:
                logger.info("Attempting to load vector store from disk...")
                vectorstore = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Successfully loaded vector store with {vectorstore.index.ntotal} documents.")
                return vectorstore
            except Exception as e:
                logger.error(f"CRITICAL: Failed to load existing vector store: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.warning("Proceeding to create a new, empty vector store.")
        
        # Create new vectorstore if loading fails or it doesn't exist
        logger.info("Creating new FAISS vectorstore")
        
        try:
            # Create an empty vectorstore without dummy documents
            from langchain_community.docstore.in_memory import InMemoryDocstore
            import faiss
            
            # Get embedding dimension
            sample_embedding = self.embeddings.embed_query("test")
            dimension = len(sample_embedding)
            
            # Create empty FAISS index
            index = faiss.IndexFlatL2(dimension)
            
            # Create empty docstore
            docstore = InMemoryDocstore({})
            
            # Create FAISS vectorstore
            vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id={}
            )
            
            logger.info("Successfully created a new, empty vector store.")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create a new vector store: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def add_documents(self, documents: List[Document], show_progress: bool = True) -> List[str]:
        """Add documents to the vector store with optimized batch processing.
        
        Args:
            documents: List of documents to add
            show_progress: Whether to show progress information
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
        
        try:
            logger.info(f"Attempting to add {len(documents)} documents to vector store")
            
            # Debug: Show first document content
            if documents:
                first_doc = documents[0]
                logger.info(f"First document content length: {len(first_doc.page_content)}")
                logger.info(f"First document metadata: {first_doc.metadata}")
                logger.info(f"First document content preview: {first_doc.page_content[:100]}...")
            
            # Debug: Check vector store state before adding
            initial_count = self.vectorstore.index.ntotal
            logger.info(f"Vector store initial count: {initial_count}")
            
            # Use batch processing for better performance
            if settings.use_local_embeddings and len(documents) > 10:
                logger.info("Using optimized batch embedding processing...")
                ids = self._add_documents_batch(documents, show_progress)
            else:
                # Standard processing for small batches
                logger.info("Using standard document processing...")
                ids = self.vectorstore.add_documents(documents)
            
            logger.info(f"Successfully added documents, received {len(ids)} IDs")
            
            # Debug: Check vector store state after adding
            final_count = self.vectorstore.index.ntotal
            logger.info(f"Vector store final count: {final_count}")
            
            # Persist the changes
            logger.info("Attempting to save vector store")
            self.save()
            logger.info("Vector store saved successfully")
            
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _add_documents_batch(self, documents: List[Document], show_progress: bool = True) -> List[str]:
        """Add documents using batch embedding processing for better performance."""
        from src.models.embeddings_model import embedding_manager
        
        # Extract texts for batch embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = embedding_manager.embed_documents_batch(
            texts, 
            batch_size=settings.embedding_batch_size,
            show_progress=show_progress
        )
        
        # Add texts with pre-computed embeddings
        logger.info("Adding documents to vector store...")
        metadatas = [doc.metadata for doc in documents]
        
        # Use add_texts which should be more efficient for batch operations
        ids = self.vectorstore.add_texts(texts, metadatas)
        
        logger.info(f"Successfully added {len(documents)} documents with batch processing")
        return ids
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        try:
            ids = self.vectorstore.add_texts(texts, metadatas or [])
            logger.info(f"Added {len(texts)} texts to vector store")
            
            # Persist the changes
            self.save()
            
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add texts to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter: Optional metadata filter
            fetch_k: Number of documents to fetch before filtering
            
        Returns:
            List of similar documents
        """
        try:
            if filter:
                docs = self.vectorstore.similarity_search(
                    query, k=k, filter=filter, fetch_k=fetch_k
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
            
            logger.debug(f"Found {len(docs)} similar documents for query: {query[:100]}...")
            return docs
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if score_threshold is not None:
                docs_with_scores = [
                    (doc, score) for doc, score in docs_with_scores 
                    if score >= score_threshold
                ]
            
            logger.debug(f"Found {len(docs_with_scores)} documents with scores for query: {query[:100]}...")
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """Get a retriever for the vector store.
        
        Args:
            search_type: Type of search ('similarity', 'similarity_score_threshold', 'mmr')
            **kwargs: Additional arguments for the retriever
            
        Returns:
            Retriever object
        """
        default_kwargs = {
            "k": settings.max_results
        }
        
        # Only add score_threshold if using similarity_score_threshold search type
        if search_type == "similarity_score_threshold":
            default_kwargs["score_threshold"] = settings.similarity_threshold
            default_kwargs["fetch_k"] = settings.max_results * 3
        elif search_type == "mmr":
            default_kwargs["fetch_k"] = settings.max_results * 3
            default_kwargs["lambda_mult"] = 0.7
        
        default_kwargs.update(kwargs)
        
        logger.info(f"Creating retriever with search_type: {search_type}")
        logger.info(f"Retriever kwargs: {default_kwargs}")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=default_kwargs
        )
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.vectorstore.delete(ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
            self.save()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        try:
            count = self.vectorstore.index.ntotal
            logger.info(f"Vector store document count: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def search_by_metadata(self, filter_dict: Dict[str, Any], k: int = 10) -> List[Document]:
        """Search documents by metadata filter.
        
        Args:
            filter_dict: Metadata filter
            k: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            # Use similarity search with metadata filtering if available
            # This is more compatible across different LangChain versions
            if hasattr(self.vectorstore, 'similarity_search') and filter_dict:
                try:
                    # Try to use metadata filtering if supported
                    results = self.vectorstore.similarity_search("", k=k, filter=filter_dict)
                    if results:
                        return results
                except Exception:
                    pass
            
            # Fallback: Get all documents and filter manually
            # This approach works by doing a broad similarity search and then filtering
            try:
                # Get more documents than needed for filtering
                broad_results = self.vectorstore.similarity_search("", k=k*5)
                filtered_docs = []
                
                for doc in broad_results:
                    match = True
                    for key, value in filter_dict.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                        if len(filtered_docs) >= k:
                            break
                
                logger.debug(f"Found {len(filtered_docs)} documents matching metadata filter")
                return filtered_docs
                
            except Exception as inner_e:
                logger.warning(f"Broad search filtering failed: {inner_e}")
                return []
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def save(self):
        """Persist the vector store to disk."""
        try:
            # Ensure directory exists
            self._ensure_directory_exists()
            
            logger.info(f"Saving vector store to {self.persist_directory}")
            self.vectorstore.save_local(self.persist_directory)
            logger.info("Vector store saved successfully")
            
            # Verify the save worked by checking if files exist
            index_path = Path(self.persist_directory) / "index.faiss"
            docstore_path = Path(self.persist_directory) / "index.pkl"
            
            if index_path.exists() and docstore_path.exists():
                logger.info(f"Verified: Vector store files exist at {self.persist_directory}")
            else:
                logger.warning(f"Warning: Vector store files not found after save at {self.persist_directory}")
                
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def clear(self):
        """Clear all documents from the vector store."""
        try:
            # Create a new empty vectorstore
            from langchain_community.docstore.in_memory import InMemoryDocstore
            import faiss
            
            # Get embedding dimension
            sample_embedding = self.embeddings.embed_query("test")
            dimension = len(sample_embedding)
            
            # Create empty FAISS index
            index = faiss.IndexFlatL2(dimension)
            
            # Create empty docstore
            docstore = InMemoryDocstore({})
            
            # Create FAISS vectorstore
            self._vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id={}
            )
            
            # Remove persisted files
            index_path = Path(self.persist_directory) / "index.faiss"
            docstore_path = Path(self.persist_directory) / "index.pkl"
            
            if index_path.exists():
                index_path.unlink()
            if docstore_path.exists():
                docstore_path.unlink()
            
            logger.info("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")


# Global vector store instance
vector_store = VectorStore()