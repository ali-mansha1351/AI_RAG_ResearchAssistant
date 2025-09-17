"""Research agent implementation for RAG-based question answering."""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json
from datetime import datetime


from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory

from src.models.llm_models import get_llm
from src.database.vector_store import vector_store
from src.utils.document_processing import DocumentProcessor
from config.settings import settings
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Container for research results."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class IngestionResult:
    """Container for document ingestion results."""
    success: bool
    documents_processed: int
    chunks_created: int
    sources: List[str]
    errors: Optional[List[str]] = None


class ResearchAgent:
    """AI Research Assistant that can ingest documents and answer questions."""
    
    def __init__(self, llm_type: str = "groq"):
        """Initialize the research agent.
        
        Args:
            llm_type: Type of LLM to use ('groq', 'openai', 'mock')
        """
        self.llm = get_llm(llm_type)
        self.vector_store = vector_store
        
        # Ensure vector store is loaded
        _ = self.vector_store.vectorstore  # This triggers loading
        
        self.document_processor = DocumentProcessor()
        
        # Initialize conversation memory (k=100 for last 100 message exchanges)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=100  # Remember last 100 exchanges (200 total messages)
        )
        
        # Track if we're using conversational mode
        self.use_conversation_memory = True
        
        # Initialize the QA chain (will setup conversational chain)
        self._setup_qa_chain()
        
        logger.info(f"Research agent initialized with LLM: {llm_type}")
        logger.info(f"Conversation memory enabled: k={100}")
        logger.info(f"Vector store has {self.vector_store.get_document_count()} documents")
    
    def _setup_qa_chain(self):
        """Setup the question-answering chain with conversational memory."""
        # Custom prompt template for conversational QA
        contextual_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

        contextual_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextual_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Setup retriever
        self.retriever = self.vector_store.get_retriever(
            search_type="similarity",
            k=settings.max_results
        )
        
        # Store retriever settings for potential reconfiguration
        self.retriever_settings = {
            "search_type": "similarity",
            "k": settings.max_results
        }
        
        if self.use_conversation_memory:
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, contextual_q_prompt
            )
            
            # Create retrieval chain with conversation history
            from langchain.chains.combine_documents import create_stuff_documents_chain
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            self.qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            logger.info("Conversational QA chain initialized with memory using new LangChain API")
        else:
            # Fallback to regular retrieval chain without memory
            simple_qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    ("human", "{input}"),
                ]
            )
            
            from langchain.chains.combine_documents import create_stuff_documents_chain
            question_answer_chain = create_stuff_documents_chain(self.llm, simple_qa_prompt)
            self.qa_chain = create_retrieval_chain(self.retriever, question_answer_chain)
            logger.info("Standard QA chain initialized without memory using new LangChain API")
    
    def reconfigure_retriever(self, search_type: str = "similarity", k: Optional[int] = None, score_threshold: Optional[float] = None):
        """Reconfigure the retriever with new settings.
        
        Args:
            search_type: Type of search
            k: Number of documents to retrieve
            score_threshold: Similarity threshold (if applicable)
        """
        self.retriever_settings.update({
            "search_type": search_type,
            "k": k or settings.max_results
        })
        
        if score_threshold is not None and search_type == "similarity_score_threshold":
            self.retriever_settings["score_threshold"] = score_threshold
        
        self.retriever = self.vector_store.get_retriever(**self.retriever_settings)
        
        # Re-setup QA chain with new retriever
        self._setup_qa_chain()
        
        logger.info(f"Retriever reconfigured: {self.retriever_settings}")
    
    def _refresh_retriever(self):
        """Refresh the retriever to ensure it uses the latest vector store state."""
        self.retriever = self.vector_store.get_retriever(**self.retriever_settings)
        # Re-setup the entire QA chain to ensure consistency
        self._setup_qa_chain()
        
        logger.info("Retriever refreshed")
    
    def ingest_documents(self, file_paths: List[str]) -> IngestionResult:
        """Ingest documents into the vector store with performance monitoring.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            IngestionResult with processing statistics
        """
        import time
        start_time = time.time()
        
        all_documents = []
        processed_sources = []
        errors = []
        
        logger.info(f"Starting ingestion of {len(file_paths)} files")
        
        for i, file_path in enumerate(file_paths):
            try:
                file_start = time.time()
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                processed_doc = self.document_processor.process_file(file_path, show_progress=True)
                chunks_created = 0
                if processed_doc.chunks:
                    all_documents.extend(processed_doc.chunks)
                    chunks_created = len(processed_doc.chunks)
                processed_sources.append(file_path)
                
                file_time = time.time() - file_start
                logger.info(f"File processed in {file_time:.2f}s: {chunks_created} chunks created")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Add documents to vector store
        if all_documents:
            try:
                embedding_start = time.time()
                logger.info(f"Adding {len(all_documents)} document chunks to vector store...")
                
                self.vector_store.add_documents(all_documents, show_progress=True)
                
                embedding_time = time.time() - embedding_start
                logger.info(f"Vector store operations completed in {embedding_time:.2f}s")
                
                # Refresh retriever to ensure it uses the updated vector store
                self._refresh_retriever()
                
            except Exception as e:
                error_msg = f"Failed to add documents to vector store: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return IngestionResult(
                    success=False,
                    documents_processed=0,
                    chunks_created=0,
                    sources=[],
                    errors=errors
                )
        
        total_time = time.time() - start_time
        logger.info(f"Total ingestion completed in {total_time:.2f}s")
        
        return IngestionResult(
            success=len(errors) == 0,
            documents_processed=len(processed_sources),
            chunks_created=len(all_documents),
            sources=processed_sources,
            errors=errors if errors else None
        )
    
    def ask_question(self, question: str, include_sources: bool = True) -> ResearchResult:
        """Ask a question and get an answer based on ingested documents.
        
        Args:
            question: Question to ask
            include_sources: Whether to include source information
            
        Returns:
            ResearchResult with answer and sources
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # For specific queries about structured data, try multiple search approaches
            if any(keyword in question.lower() for keyword in ['qr code', 'fields', 'required', 'mandatory', 'data elements']):
                logger.info("Using enhanced search for structured data query")
                try:
                    # Use a broader search for structured data queries
                    expanded_queries = [
                        question,
                        f"QR code {question}",
                        "mandatory data elements QR code",
                        "required fields QR code"
                    ]
                    
                    all_docs = []
                    for query in expanded_queries:
                        docs = self.retriever.get_relevant_documents(query)
                        all_docs.extend(docs)
                    
                    # Remove duplicates based on content
                    unique_docs = []
                    seen_content = set()
                    for doc in all_docs:
                        content_hash = hash(doc.page_content[:100])
                        if content_hash not in seen_content:
                            unique_docs.append(doc)
                            seen_content.add(content_hash)
                    
                    # Limit to reasonable number
                    source_documents = unique_docs[:settings.max_results * 2]
                    logger.info(f"Enhanced search found {len(source_documents)} relevant documents")
                    
                    if source_documents:
                        # Create context manually for better control
                        context = "\n\n".join([doc.page_content for doc in source_documents])
                        
                        # Use LLM directly with enhanced context
                        enhanced_prompt = f"""Use the following context to answer the question about QR code fields and requirements.

Context:
{context}

Question: {question}

Answer: Based on the provided context, let me identify the specific QR code requirements:"""

                        llm_response = self.llm.invoke(enhanced_prompt)
                        # Handle different response types safely
                        try:
                            # Try common attributes
                            answer_text = getattr(llm_response, 'content', None) or \
                                         getattr(llm_response, 'text', None) or \
                                         str(llm_response)
                        except Exception:
                            answer_text = str(llm_response)
                        
                        result = {"answer": answer_text, "context": source_documents}
                    else:
                        # Fallback to normal QA chain if no documents found
                        logger.warning("Enhanced search found no documents, falling back to normal search")
                        if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                            logger.warning("QA chain not initialized, recreating...")
                            self._setup_qa_chain()
                        
                        if self.use_conversation_memory:
                            chat_history = self.memory.chat_memory.messages
                            result = self.qa_chain.invoke({
                                "input": question,
                                "chat_history": chat_history
                            })
                        else:
                            result = self.qa_chain.invoke({"input": question})
                        
                        # Handle result safely
                        if not isinstance(result, dict):
                            logger.warning(f"Unexpected result type in fallback: {type(result)}")
                            result = {"answer": str(result), "context": []}
                        
                except Exception as enhanced_error:
                    logger.error(f"Enhanced search failed: {enhanced_error}")
                    # Fallback to normal QA chain
                    if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                        logger.warning("QA chain not initialized, recreating...")
                        self._setup_qa_chain()
                    
                    if self.use_conversation_memory:
                        chat_history = self.memory.chat_memory.messages
                        result = self.qa_chain.invoke({
                            "input": question,
                            "chat_history": chat_history
                        })
                    else:
                        result = self.qa_chain.invoke({"input": question})
                    
                    # Handle result safely
                    if not isinstance(result, dict):
                        logger.warning(f"Unexpected result type in enhanced fallback: {type(result)}")
                        result = {"answer": str(result), "context": []}
                
            else:
                # Ensure QA chain is available
                if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                    logger.warning("QA chain not initialized, recreating...")
                    self._setup_qa_chain()
                
                # Debug: Test retriever directly
                logger.info("Testing retriever with question...")
                test_docs = self.retriever.get_relevant_documents(question)
                logger.info(f"Retriever returned {len(test_docs)} documents")
                if test_docs:
                    logger.info(f"First retrieved document preview: {test_docs[0].page_content[:200]}...")
                
            # Get answer from QA chain
            if self.use_conversation_memory:
                # Use new API with conversation history
                chat_history = self.memory.chat_memory.messages
                result = self.qa_chain.invoke({
                    "input": question,
                    "chat_history": chat_history
                })
            else:
                # Use new API without conversation history
                result = self.qa_chain.invoke({"input": question})
            
            # Debug: Log the type and structure of result
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Handle different result types safely
            if isinstance(result, dict):
                answer = result.get("answer", "")
                source_documents = result.get("context", [])
            else:
                # If result is not a dict, try to extract content
                logger.warning(f"Unexpected result type: {type(result)}")
                answer = str(result)
                source_documents = []
            
            logger.info(f"QA chain returned answer: {answer[:200]}...")
            logger.info(f"Number of source documents: {len(source_documents)}")
            
            # If no source documents were retrieved, provide a helpful message
            if not source_documents:
                answer = "I couldn't find any relevant information in your documents to answer this question. This could be because:\n\n• The question is not related to the content of your uploaded documents\n• The similarity threshold is too restrictive\n• The documents haven't been processed yet\n\nTry rephrasing your question or check that your documents contain information relevant to your query."
            
            # Process sources
            sources = []
            if include_sources and source_documents:
                sources = self._process_sources(source_documents)
            
            # Calculate confidence score (simplified)
            confidence_score = self._calculate_confidence(source_documents, answer)
            
            # Manually save to conversation memory for new API
            if self.use_conversation_memory and hasattr(self, 'memory'):
                from langchain.schema import HumanMessage, AIMessage
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)
            
            return ResearchResult(
                question=question,
                answer=answer,
                sources=sources,
                confidence_score=confidence_score,
                timestamp=datetime.now(),
                metadata={
                    "num_sources": len(source_documents),
                    "llm_type": self.llm._llm_type,
                    "conversational": self.use_conversation_memory,
                    "memory_messages": len(self.memory.chat_memory.messages) if self.use_conversation_memory else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            return ResearchResult(
                question=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e), "conversational": self.use_conversation_memory}
            )
    
    def _process_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Process source documents into structured format."""
        sources = []
        
        for i, doc in enumerate(source_documents):
            source_info = {
                "id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            
            # Extract key information from metadata
            if "source" in doc.metadata:
                source_info["source"] = doc.metadata["source"]
            if "title" in doc.metadata:
                source_info["title"] = doc.metadata["title"]
            if "url" in doc.metadata:
                source_info["url"] = doc.metadata["url"]
            if "page" in doc.metadata:
                source_info["page"] = doc.metadata["page"]
                
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(self, source_documents: List[Document], answer: str) -> float:
        """Calculate confidence score based on available sources and answer quality."""
        if not source_documents:
            return 0.0
        
        # Simple confidence calculation
        base_score = min(len(source_documents) * 0.2, 1.0)  # More sources = higher confidence
        
        # Reduce confidence if answer is very short or contains uncertainty phrases
        uncertainty_phrases = ["i don't know", "not sure", "cannot determine", "unclear"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            base_score *= 0.5
        
        if len(answer.split()) < 10:  # Very short answers
            base_score *= 0.7
        
        return round(base_score, 2)
    
    def generate_summary(self, topic: str, max_sources: int = 10) -> ResearchResult:
        """Generate a comprehensive summary on a topic.
        
        Args:
            topic: Topic to summarize
            max_sources: Maximum number of sources to use
            
        Returns:
            ResearchResult with summary
        """
        # Search for relevant documents
        relevant_docs = self.vector_store.similarity_search(topic, k=max_sources)
        
        if not relevant_docs:
            return ResearchResult(
                question=f"Summary of: {topic}",
                answer="I don't have enough information in my knowledge base to provide a comprehensive summary on this topic.",
                sources=[],
                confidence_score=0.0,
                timestamp=datetime.now()
            )
        
        # Create a comprehensive prompt for summary generation
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        summary_question = f"Please provide a comprehensive summary about {topic} based on the available information. Include key points, important details, and any relevant insights."
        
        # Use the QA chain but with summary-focused question
        try:
            if self.use_conversation_memory:
                chat_history = self.memory.chat_memory.messages
                result = self.qa_chain.invoke({
                    "input": summary_question,
                    "chat_history": chat_history
                })
            else:
                result = self.qa_chain.invoke({"input": summary_question})
            
            # Handle result safely
            if isinstance(result, dict):
                answer = result.get("answer", "")
            else:
                logger.warning(f"Unexpected result type in summary: {type(result)}")
                answer = str(result)
            
            return ResearchResult(
                question=f"Summary of: {topic}",
                answer=answer,
                sources=self._process_sources(relevant_docs),
                confidence_score=self._calculate_confidence(relevant_docs, answer),
                timestamp=datetime.now(),
                metadata={
                    "summary_type": "comprehensive",
                    "num_sources": len(relevant_docs)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return ResearchResult(
                question=f"Summary of: {topic}",
                answer=f"I encountered an error while generating the summary: {str(e)}",
                sources=[],
                confidence_score=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the current knowledge base."""
        try:
            doc_count = self.vector_store.get_document_count()
            
            # Test retrieval with a simple query
            test_query = "test"
            test_docs = self.retriever.get_relevant_documents(test_query)
            logger.info(f"Test retrieval returned {len(test_docs)} documents")
            
            # Get sample documents to analyze sources
            sample_docs = self.vector_store.similarity_search("", k=min(100, doc_count))
            
            sources = set()
            document_types = {}
            
            for doc in sample_docs:
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
                if "type" in doc.metadata:
                    doc_type = doc.metadata["type"]
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1
            
            return {
                "total_chunks": doc_count,
                "unique_sources": len(sources),
                "document_types": document_types,
                "sample_sources": list(sources)[:10],  # First 10 sources
                "test_retrieval_count": len(test_docs)
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"error": str(e)}
    
    def clear_conversation_memory(self) -> bool:
        """Clear conversation memory to start a new chat session."""
        try:
            if self.use_conversation_memory and hasattr(self, 'memory'):
                self.memory.clear()
                logger.info("Conversation memory cleared")
                return True
            else:
                logger.info("Conversation memory not enabled or available")
                return False
        except Exception as e:
            logger.error(f"Failed to clear conversation memory: {e}")
            return False
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        try:
            if not self.use_conversation_memory or not hasattr(self, 'memory'):
                return {"enabled": False, "message": "Conversation memory not enabled"}
            
            messages = self.memory.chat_memory.messages
            return {
                "enabled": True,
                "total_messages": len(messages),
                "conversation_exchanges": len(messages) // 2,
                "memory_window": self.memory.k,
                "last_messages": [
                    {
                        "type": msg.type if hasattr(msg, 'type') else "unknown",
                        "content": str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    }
                    for msg in messages[-6:]  # Last 6 messages (3 exchanges)
                ] if messages else []
            }
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {"error": str(e)}
    
    def toggle_conversation_mode(self, enable: Optional[bool] = None) -> bool:
        """Toggle conversation mode on/off."""
        try:
            if enable is None:
                # Toggle current state
                self.use_conversation_memory = not self.use_conversation_memory
            else:
                self.use_conversation_memory = enable
            
            # Re-setup QA chain with new mode
            self._setup_qa_chain()
            
            logger.info(f"Conversation mode {'enabled' if self.use_conversation_memory else 'disabled'}")
            return self.use_conversation_memory
        except Exception as e:
            logger.error(f"Failed to toggle conversation mode: {e}")
            return self.use_conversation_memory
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base."""
        try:
            self.vector_store.clear()
            logger.info("Knowledge base cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            return False