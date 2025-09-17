"""Streamlit frontend for the AI Research Assistant."""

import streamlit as st
import os
import sys
from pathlib import Path
import json
import tempfile
from typing import List
import logging
from pydantic import ValidationError
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.agents.research_agents import ResearchAgent, ResearchResult
from src.database.vector_store import vector_store

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Applies custom CSS to the Streamlit application."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    :root {
        --bg-color: #0a0a0b;
        --primary-accent: #0066CC;
        --secondary-accent: #8A2BE2; /* Purple */
        --text-color: #EAEAEA;
        --glass-bg: rgba(26, 26, 27, 0.6);
        --border-color: rgba(255, 255, 255, 0.1);
        --font-main: 'Inter', sans-serif;
        --font-mono: 'IBM Plex Mono', monospace;
    }

    /* General Body and Background */
    body {
        font-family: var(--font-main);
        color: var(--text-color);
    }

    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 27% 37%, hsla(215, 98%, 62%, 0.1) 0px, transparent 50%),
            radial-gradient(at 97% 21%, hsla(286, 98%, 62%, 0.1) 0px, transparent 50%),
            radial-gradient(at 52% 99%, hsla(355, 98%, 62%, 0.1) 0px, transparent 50%),
            radial-gradient(at 10% 29%, hsla(256, 96%, 61%, 0.1) 0px, transparent 50%),
            radial-gradient(at 97% 96%, hsla(38, 60%, 62%, 0.1) 0px, transparent 50%),
            radial-gradient(at 33% 50%, hsla(222, 67%, 73%, 0.1) 0px, transparent 50%),
            radial-gradient(at 79% 53%, hsla(343, 68%, 63%, 0.1) 0px, transparent 50%);
        background-attachment: fixed;
        animation: gradient-animation 15s ease infinite;
    }

    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar Styling */
    .st-emotion-cache-16txtl3 {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid var(--border-color);
    }

    .st-emotion-cache-16txtl3 .st-emotion-cache-1v0mbdj, .st-emotion-cache-16txtl3 .st-emotion-cache-1v0mbdj p {
        color: var(--text-color);
    }
    
    .st-emotion-cache-16txtl3 h2 {
        color: var(--text-color);
        font-family: var(--font-main);
        text-align: center;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1rem;
    }

    /* Main Content Area */
    .st-emotion-cache-z5fcl4 {
        padding-top: 2rem;
    }

    /* Header and API Status */
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: rgba(10, 10, 11, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-color);
    }
    
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 0;
    }

    .api-status {
        display: flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .api-status-success {
        background-color: rgba(0, 204, 102, 0.15);
        color: #00CC66;
    }
    
    .api-status-success::before {
        content: '●';
        color: #00CC66;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    .api-status-failure {
        background-color: rgba(255, 71, 87, 0.15);
        color: #FF4757;
    }

    .api-status-failure::before {
        content: '●';
        color: #FF4757;
        margin-right: 8px;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border: none;
        color: var(--text-color) !important;
        opacity: 0.6;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: transparent;
        border-bottom: 2px solid var(--primary-accent);
        opacity: 1;
    }

    /* Input and Button Styling */
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background: var(--glass-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(5px);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(138, 43, 226, 0.3);
    }

    /* File Uploader */
    .stFileUploader > div {
        border: 2px dashed var(--border-color);
        background: var(--glass-bg);
        border-radius: 8px;
    }

    /* AI Response and Chat Styling */
    .ai-response-container {
        background: var(--glass-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-top: 1rem;
    }
    
    .ai-response-container h3 {
        font-family: var(--font-main);
        color: var(--primary-accent);
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
    }

    .ai-response-container p, .ai-response-container li {
        font-family: var(--font-main);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .ai-response-container code {
        font-family: var(--font-mono);
        background: rgba(0,0,0,0.3);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }

    /* Source and Stats Boxes */
    .source-box, .stats-box {
        background: var(--glass-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .source-box:hover, .stats-box:hover {
        border-color: var(--primary-accent);
        transform: translateY(-3px);
    }
    .source-box h4 {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--primary-accent);
        margin-bottom: 0.5rem;
    }
    .source-box p {
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.8;
    }

    /* Custom Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit's default header and footer */
    .st-emotion-cache-18ni7ap, .st-emotion-cache-h4y42s {
        display: none;
    }
    
    /* Adjust main content padding for fixed header */
    .st-emotion-cache-z5fcl4 {
        padding-top: 6rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def check_groq_api_key():
    """Checks if the Groq API key is available."""
    try:
        # The Settings class will raise a validation error if the key is missing
        Settings()
        return True
    except ValidationError:
        return False

def get_or_create_agent():
    """Get or create a persistent research agent using session state."""
    if 'research_agent' not in st.session_state:
        st.session_state.research_agent = ResearchAgent()
    return st.session_state.research_agent

def initialize_chat_history():
    """Initialize chat history in session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def save_to_chat_history(question: str, result: ResearchResult):
    """Save question and answer to chat history."""
    exchange = {
        "question": question,
        "answer": result.answer,
        "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": len(result.sources),
        "confidence": result.confidence_score
    }
    st.session_state.chat_history.append(exchange)
    
    # Keep only last 20 exchanges to prevent memory bloat
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

def clear_chat_history():
    """Clear chat history and conversation memory."""
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []
    
    # Clear agent conversation memory too
    agent = get_or_create_agent()
    agent.clear_conversation_memory()
    
    # Clear current result
    if 'result' in st.session_state:
        del st.session_state.result
    if 'last_question' in st.session_state:
        del st.session_state.last_question

def display_chat_history():
    """Display previous chat exchanges."""
    if st.session_state.chat_history:
        st.subheader("💬 Recent Conversation")
        
        # Show conversation summary
        agent = get_or_create_agent()
        conv_summary = agent.get_conversation_summary()
        
        if conv_summary.get("enabled", False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💭 Exchanges", conv_summary.get("conversation_exchanges", 0))
            with col2:
                st.metric("📝 Messages", conv_summary.get("total_messages", 0))
            with col3:
                st.metric("🧠 Memory Window", conv_summary.get("memory_window", 0))
        
        # Display recent conversations in reverse order (newest first)
        for i, exchange in enumerate(reversed(st.session_state.chat_history[-10:])):  # Last 10
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {exchange['question'][:60]}..." if len(exchange['question']) > 60 else f"Q{len(st.session_state.chat_history)-i}: {exchange['question']}"):
                st.markdown(f"**🤔 Question:** {exchange['question']}")
                st.markdown(f"**🤖 Answer:** {exchange['answer']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏰ {exchange['timestamp']}")
                with col2:
                    st.caption(f"📚 {exchange['sources']} sources")
                with col3:
                    st.caption(f"🎯 {exchange['confidence']*100:.0f}% confidence")
    else:
        st.info("💬 Start a conversation by asking a question!")

def enhanced_chat_interface():
    """Enhanced chat interface with conversation memory."""
    # Initialize chat history
    initialize_chat_history()
    
    # Chat controls row
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.header("💬 Chat with Your Documents")
    
    with col2:
        # New Chat button
        if st.button("🆕 New Chat", help="Start a new conversation (clears memory)", key="new_chat_btn"):
            clear_chat_history()
            st.success("🔄 Started new conversation!")
            st.rerun()
    
    with col3:
        # Memory toggle
        agent = get_or_create_agent()
        memory_enabled = agent.use_conversation_memory
        
        if st.button(f"🧠 {'ON' if memory_enabled else 'OFF'}", help=f"Conversation memory: {'ON' if memory_enabled else 'OFF'}", key="memory_toggle"):
            new_state = agent.toggle_conversation_mode()
            st.success(f"💭 Conversation memory {'enabled' if new_state else 'disabled'}!")
            st.rerun()
    
    with col4:
        # Chat history toggle
        show_history = st.checkbox("📋 History", help="Show conversation history", key="history_toggle")
    
    # Display conversation status
    agent = get_or_create_agent()
    conv_summary = agent.get_conversation_summary()
    if conv_summary.get("enabled", False) and conv_summary.get("total_messages", 0) > 0:
        st.info(f"💭 Active conversation with {conv_summary.get('conversation_exchanges', 0)} exchanges | Memory window: {conv_summary.get('memory_window', 0)} exchanges")
    
    # Chat history section
    if show_history and st.session_state.chat_history:
        with st.expander("📋 Conversation History", expanded=True):
            display_chat_history()
    
    return agent

def process_documents_with_progress(agent, file_paths, uploaded_files, main_progress, main_status, file_progress_container, steps_container):
    """Process documents with detailed progress tracking."""
    import time
    
    with steps_container:
        st.write("**Processing Steps:**")
        
        # Step 2: Process each file (20-70% of progress)
        all_documents = []
        processed_sources = []
        errors = []
        
        for i, file_path in enumerate(file_paths):
            try:
                start_time = time.time()
                
                # Update progress
                file_progress = 0.1 + (i / len(file_paths)) * 0.6
                main_progress.progress(file_progress)
                
                with file_progress_container:
                    st.text(f"📄 Processing: {uploaded_files[i].name}")
                
                # Process file
                processed_doc = agent.document_processor.process_file(file_path, show_progress=False)
                
                if processed_doc.chunks:
                    all_documents.extend(processed_doc.chunks)
                    processed_sources.append(file_path)
                    
                    processing_time = time.time() - start_time
                    
                    st.write(f"✅ {uploaded_files[i].name}: {len(processed_doc.chunks)} chunks ({processing_time:.1f}s)")
                else:
                    st.write(f"⚠️ {uploaded_files[i].name}: No content extracted")
                    
            except Exception as e:
                error_msg = f"Failed to process {uploaded_files[i].name}: {str(e)}"
                errors.append(error_msg)
                st.write(f"❌ {uploaded_files[i].name}: Error - {str(e)}")
        
        # Step 3: Generate embeddings (70-90% of progress)
        main_progress.progress(0.7)
        main_status.text("Step 3/4: Generating embeddings...")
        
        with file_progress_container:
            st.text(f"🧠 Generating embeddings for {len(all_documents)} chunks...")
        
        if all_documents:
            try:
                embedding_start = time.time()
                agent.vector_store.add_documents(all_documents, show_progress=True)
                embedding_time = time.time() - embedding_start
                
                st.write(f"✅ Embeddings generated: {len(all_documents)} chunks ({embedding_time:.1f}s)")
                
                # Refresh retriever
                agent._refresh_retriever()
                
            except Exception as e:
                error_msg = f"Failed to add documents to vector store: {str(e)}"
                errors.append(error_msg)
                st.write(f"❌ Vector store error: {str(e)}")
        
        # Step 4: Finalize (90-100% of progress)
        main_progress.progress(0.9)
        main_status.text("Step 4/4: Finalizing...")
        
        with file_progress_container:
            st.text("💾 Saving to disk...")
        
        time.sleep(0.5)  # Brief pause for visual feedback
        
    # Return result
    from src.agents.research_agents import IngestionResult
    return IngestionResult(
        success=len(errors) == 0,
        documents_processed=len(processed_sources),
        chunks_created=len(all_documents),
        sources=processed_sources,
        errors=errors if errors else None
    )

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    apply_custom_css()

    # --- Header ---
    api_key_available = check_groq_api_key()
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🤖 AI Research Assistant</h1>
            <div class="api-status {'api-status-success' if api_key_available else 'api-status-failure'}">
                {'API Online' if api_key_available else 'API Key Missing'}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.markdown("## 🛠️ Controls")
        
        if not api_key_available:
            st.error("Groq API key is not set. Please add it to your .env file.")
            st.stop()

        # Knowledge Base Status using persistent agent
        st.markdown("### 📚 Knowledge Base Status")
        try:
            agent = get_or_create_agent()
            # Check raw vector count instead of similarity search
            vector_count = agent.vector_store.vectorstore.index.ntotal
            if vector_count > 0:
                st.success(f"✅ {vector_count} document chunks available")
            else:
                st.info("📭 No documents uploaded yet")
        except Exception as e:
            st.warning(f"⚠️ Knowledge base status unclear: {str(e)[:50]}...")

        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload documents to be processed and stored in the vector database."
        )

        if uploaded_files:
            # Check for duplicate files
            try:
                stats = agent.get_knowledge_base_stats()
                existing_sources = set(stats.get('sample_sources', []))
            except:
                existing_sources = set()
            
            # Categorize files
            new_files = []
            skipped_files = []
            
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                # Check if file already exists in knowledge base
                if any(file_name in str(source) for source in existing_sources):
                    skipped_files.append(file_name)
                else:
                    new_files.append(uploaded_file)
            
            # Show file status
            if skipped_files:
                st.info(f"📄 **Already processed:** {', '.join(skipped_files[:3])}" + 
                       (f" and {len(skipped_files)-3} more" if len(skipped_files) > 3 else ""))
            
            if new_files:
                st.success(f"📄 **New files to process:** {len(new_files)} documents")
                
                # Show file size warnings
                large_files = [f for f in new_files if f.size > 2 * 1024 * 1024]  # > 2MB
                if large_files:
                    st.warning(f"⚠️ **Large files detected:** {len(large_files)} files > 2MB. Processing may take longer.")
            
            if st.button("Process Documents", disabled=len(new_files) == 0):
                if len(new_files) == 0:
                    st.warning("No new files to process!")
                else:
                    # Process with detailed progress tracking
                    progress_container = st.container()
                    
                    with progress_container:
                        # Main progress bar
                        main_progress = st.progress(0)
                        main_status = st.empty()
                        
                        # File-by-file progress
                        file_progress_container = st.container()
                        
                        # Detailed steps
                        steps_container = st.expander("📊 Processing Details", expanded=True)
                    
                    try:
                        agent = get_or_create_agent()
                        
                        # Step 1: Save files (10% of progress)
                        main_status.text("Step 1/4: Saving uploaded files...")
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []
                        
                        total_files = len(new_files)
                        for i, uploaded_file in enumerate(new_files):
                            file_progress = (i / total_files) * 0.1
                            main_progress.progress(file_progress)
                            
                            with file_progress_container:
                                st.text(f"💾 Saving: {uploaded_file.name}")
                            
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(file_path)
                        
                        main_progress.progress(0.1)
                        main_status.text("Step 2/4: Processing documents...")
                        
                        # Step 2: Process documents with enhanced progress tracking
                        result = process_documents_with_progress(
                            agent, file_paths, new_files, 
                            main_progress, main_status, 
                            file_progress_container, steps_container
                        )
                        
                        # Step 4: Complete processing
                        main_progress.progress(1.0)
                        main_status.text("✅ Processing complete!")
                        
                        # Clear file progress
                        with file_progress_container:
                            st.text("🎉 All done!")
                        
                        # Show results summary
                        with steps_container:
                            st.write("---")
                            st.write("**📊 Summary:**")
                            st.write(f"- Files processed: {result.documents_processed}")
                            st.write(f"- Chunks created: {result.chunks_created}")
                            if result.errors:
                                st.write(f"- Errors: {len(result.errors)}")
                        
                        # Test retrieval immediately after processing
                        try:
                            # Check raw vector counts instead of similarity search
                            st.write("---")
                            st.write("🔄 **Verifying Persistence...**")
                            
                            # Agent's current vector store count
                            agent_count = agent.vector_store.vectorstore.index.ntotal
                            st.write(f"- **Raw vector count in agent's current instance:** `{agent_count}`")
                            
                            # Fresh instance count
                            from src.database.vector_store import VectorStore
                            fresh_vector_store = VectorStore()
                            fresh_count = fresh_vector_store.vectorstore.index.ntotal
                            st.write(f"- **Raw vector count in fresh vector store instance:** `{fresh_count}`")
                            st.write("---")
                            
                        except Exception as e:
                            st.write(f"- Vector store check error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                        
                        if result.success:
                            st.success("✅ Documents processed and embedded successfully!")
                            st.info(f"📊 Processed {result.documents_processed} files, created {result.chunks_created} text chunks")
                            
                            # Show processed files
                            with st.expander("📁 Processed Files"):
                                for i, source in enumerate(result.sources):
                                    st.write(f"• {new_files[i].name}")
                        else:
                            st.error("❌ Some documents failed to process")
                            if result.errors:
                                with st.expander("🔍 Error Details"):
                                    for error in result.errors:
                                        st.error(f"• {error}")
                        
                        # Clean up temporary files
                        for file_path in file_paths:
                            os.remove(file_path)
                        os.rmdir(temp_dir)

                    except Exception as e:
                        st.error(f"An error occurred during document processing: {e}")
                        logger.error(f"Document processing error: {e}")
                        
                        # Show detailed error for debugging
                        with st.expander("🔍 Debug Information"):
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("📄 All uploaded files have already been processed.")

    # --- Main Content Area ---
    
    # About Section
    st.markdown(
        """
        <div class="ai-response-container">
            <h3>📖 About This AI Research Assistant</h3>
            <p>This intelligent assistant uses Retrieval-Augmented Generation (RAG) to provide accurate, 
            source-backed answers from your uploaded documents. Upload your files, ask questions, and get 
            AI-powered insights with confidence scores and source citations.</p>
            <p><strong>Technologies:</strong> Streamlit • LangChain • Groq API • FAISS Vector Database</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "📝 Summary", "📊 Sources & Stats"])

    # Use persistent agent throughout the app
    agent = get_or_create_agent()

    with tab1:
        # Enhanced chat interface with memory
        agent = enhanced_chat_interface()
        
        question = st.text_area(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings? (I remember our conversation!)",
            height=100,
            key="question_input"
        )

        col1, col2 = st.columns([3, 1])
        
        with col1:
            ask_button = st.button("🔍 Ask Question", type="primary", key="get_answer_button")
        
        with col2:
            follow_up = st.button("🔄 Follow Up", help="Ask a follow-up question", key="follow_up_btn")

        if (ask_button or follow_up) and question:
            with st.spinner("🧠 AI is thinking..."):
                try:
                    # Check if there are any documents in the vector store first
                    vector_count = agent.vector_store.vectorstore.index.ntotal
                    st.info(f"📚 Found {vector_count} document chunks in knowledge base")
                    
                    if vector_count == 0:
                        st.warning("⚠️ No documents found in knowledge base. Please upload and process documents first.")
                        return
                    
                    # Test retrieval
                    test_docs = agent.retriever.get_relevant_documents(question)
                    st.info(f"🔍 Test retrieval found {len(test_docs)} relevant documents")
                    
                    result = agent.ask_question(question)
                    st.session_state.result = result
                    st.session_state.last_question = question
                    
                    # Save to chat history
                    save_to_chat_history(question, result)
                    
                    # Show debug info
                    with st.expander("🔍 Debug Information"):
                        st.write(f"**Question:** {question}")
                        st.write(f"**Vector Count:** {vector_count}")
                        st.write(f"**Retrieved Sources:** {len(result.sources)}")
                        st.write(f"**Confidence Score:** {result.confidence_score}")
                        st.write(f"**Retriever Settings:** {agent.retriever_settings}")
                        if result.metadata:
                            st.write(f"**Metadata:** {result.metadata}")
                            if 'conversational' in result.metadata:
                                st.write(f"**Conversation Mode:** {'Enabled' if result.metadata['conversational'] else 'Disabled'}")
                            if 'memory_messages' in result.metadata:
                                st.write(f"**Memory Messages:** {result.metadata['memory_messages']}")
                    
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    # Still save failed attempts to chat history
                    failed_result = ResearchResult(
                        question=question,
                        answer=f"Error: {str(e)}",
                        sources=[],
                        confidence_score=0.0,
                        timestamp=datetime.now(),
                        metadata={"error": str(e)}
                    )
                    save_to_chat_history(question, failed_result)
                    logger.error(f"Error in ask_question: {e}")
                    
                    # Show detailed error for debugging
                    with st.expander("🔍 Debug Information"):
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("Please enter a question.")

        if 'result' in st.session_state:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="ai-response-container">
                    <h3>🤖 AI's Answer</h3>
                    <p><strong>Your Question:</strong> {st.session_state.last_question}</p>
                    <p>{st.session_state.result.answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        st.header("Content Summary")
        if 'result' in st.session_state:
            # Generate summary based on the last question
            if st.button("Generate Summary", key="generate_summary_button"):
                with st.spinner("📝 Generating summary..."):
                    try:
                        summary_result = agent.generate_summary(st.session_state.last_question)
                        st.session_state.summary_result = summary_result
                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}")
            
            if 'summary_result' in st.session_state:
                st.markdown(
                    f"""
                    <div class="ai-response-container">
                        <h3>Summary of Findings</h3>
                        <p>{st.session_state.summary_result.answer}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Click 'Generate Summary' to create a summary based on your question.")
        else:
            st.info("Ask a question in the first tab to generate a summary.")

    with tab3:
        st.header("Sources and Confidence Score")
        if 'result' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="stats-box">
                        <h4>Confidence Score</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: var(--primary-accent);">
                            {getattr(st.session_state.result, 'confidence_score', 'N/A')}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            st.subheader("Cited Sources")
            sources = getattr(st.session_state.result, 'sources', [])
            if sources:
                for source in sources:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <h4>Source: {source.get('source', 'Unknown')}</h4>
                            <p>{source.get('content', 'No content available.')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No sources were cited for this answer.")
            
            # Show knowledge base stats
            st.markdown("---")
            st.subheader("Knowledge Base Statistics")
            try:
                stats = agent.get_knowledge_base_stats()
                if "error" not in stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", stats.get("total_chunks", 0))
                    with col2:
                        st.metric("Unique Sources", stats.get("unique_sources", 0))
                    with col3:
                        st.metric("Test Retrieval", stats.get("test_retrieval_count", 0))
                    
                    if stats.get("document_types"):
                        st.subheader("Document Types")
                        for doc_type, count in stats.get("document_types", {}).items():
                            st.write(f"• {doc_type}: {count}")
                else:
                    st.error(f"Error getting stats: {stats['error']}")
            except Exception as e:
                st.error(f"Failed to get knowledge base statistics: {e}")
        else:
            st.info("Ask a question in the first tab to see sources and statistics.")

if __name__ == "__main__":
    main()