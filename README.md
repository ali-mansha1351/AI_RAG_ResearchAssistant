# 🤖 AI RAG Research Assistant

A powerful AI-powered Research Assistant that combines Retrieval-Augmented Generation (RAG) with conversational memory to help you analyze documents and answer questions intelligently.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🔍 **Intelligent Document Analysis**

- **Multi-format Support**: Process PDFs, text files, and other document formats
- **Smart Chunking**: Intelligent document splitting for optimal retrieval
- **Table Extraction**: Extract structured data from PDF tables
- **Progress Tracking**: Real-time progress bars during document processing

### 💬 **Conversational AI**

- **Chat Memory**: Remembers conversation context (up to 100 exchanges)
- **Follow-up Questions**: Ask contextual follow-up questions
- **New Chat Sessions**: Start fresh conversations anytime
- **Memory Toggle**: Enable/disable conversation memory

### 🚀 **Performance Optimized**

- **Batch Processing**: Efficient document embedding in batches
- **Local Embeddings**: Use SentenceTransformers for faster processing
- **File Deduplication**: Avoid reprocessing duplicate documents
- **GPU/CPU Support**: Flexible deployment options

### 🎯 **Advanced Retrieval**

- **Multiple Search Strategies**: Similarity search, MMR, score thresholding
- **Enhanced QR Code Analysis**: Specialized processing for structured data
- **Source Attribution**: Track answer sources with confidence scores
- **Knowledge Base Stats**: Monitor document collection metrics

## 🏗️ Architecture

```
├── 📁 src/
│   ├── agents/           # Research agents and conversation logic
│   ├── database/         # Vector store management (FAISS)
│   ├── models/           # LLM and embedding model wrappers
│   └── utils/            # Document processing utilities
├── 📁 data/
│   ├── raw/             # Raw document storage
│   ├── processed/       # Processed document chunks
│   └── vector_store/    # FAISS vector database
├── 📁 config/           # Configuration settings
├── 📁 logs/             # Application logs
└── streamlit_app.py     # Main web application
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/ali-mansha1351/AI_RAG_ResearchAssistant.git
   cd AI_RAG_ResearchAssistant
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional packages**

   ```bash
   pip install sentence-transformers langchain-community
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📖 Usage

### Basic Workflow

1. **Upload Documents**

   - Click "📤 Upload Documents" in the sidebar
   - Select PDF files or drag-and-drop
   - Watch real-time progress bars during processing

2. **Ask Questions**

   - Use the chat interface to ask questions about your documents
   - The AI remembers conversation context for follow-up questions
   - View source attribution and confidence scores

3. **Manage Conversations**
   - Click "🆕 New Chat" to start fresh conversations
   - Toggle "🧠 Memory" to enable/disable conversation memory
   - View "📋 History" to see previous exchanges

### Advanced Features

#### Document Processing

```python
from src.agents.research_agents import ResearchAgent

agent = ResearchAgent()
result = agent.ingest_documents(["document1.pdf", "document2.pdf"])
print(f"Processed {result.documents_processed} documents")
```

#### Question Answering

```python
result = agent.ask_question("What are the main QR code requirements?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_score}")
```

#### Conversation Management

```python
# Clear conversation memory
agent.clear_conversation_memory()

# Toggle conversation mode
agent.toggle_conversation_mode(enable=True)

# Get conversation summary
summary = agent.get_conversation_summary()
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
MAX_RESULTS=5
BATCH_SIZE=32
SIMILARITY_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Settings Configuration

Modify `config/settings.py` to customize:

- **LLM Settings**: Choose between Groq, OpenAI, or mock LLM
- **Embedding Models**: Configure local vs API-based embeddings
- **Vector Store**: FAISS configuration and parameters
- **Document Processing**: Chunk size, overlap, and processing options

## 🔧 API Reference

### ResearchAgent Class

#### Methods

- `ingest_documents(file_paths)`: Process and store documents
- `ask_question(question)`: Ask questions with conversation memory
- `clear_conversation_memory()`: Reset conversation history
- `toggle_conversation_mode(enable)`: Enable/disable memory
- `get_conversation_summary()`: Get chat statistics
- `get_knowledge_base_stats()`: Get document collection metrics

#### Configuration Options

- `llm_type`: "groq", "openai", or "mock"
- `use_conversation_memory`: Enable conversation context
- `memory_window`: Number of exchanges to remember (default: 100)

## 📊 Performance Tips

### Optimization Strategies

1. **Use Local Embeddings**

   - Switch to SentenceTransformers for faster processing
   - Reduces API calls and improves speed

2. **Batch Processing**

   - Process documents in batches of 32 for optimal performance
   - Monitor memory usage during large document processing

3. **File Deduplication**

   - The system automatically detects duplicate files
   - Avoid reprocessing the same documents

4. **Memory Management**
   - Clear conversation memory for long sessions
   - Monitor vector store size for performance

### Hardware Recommendations

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for large document collections
- **Storage**: SSD recommended for vector database
- **GPU**: Optional, but improves embedding speed

## 🐛 Troubleshooting

### Common Issues

**"Context not provided" Error**

- Ensure documents are uploaded and processed
- Check vector store has documents loaded
- Verify similarity threshold settings

**Slow Processing**

- Switch to local embeddings (SentenceTransformers)
- Reduce batch size if memory issues occur
- Check available system resources

**Memory Errors**

- Clear conversation memory periodically
- Reduce document chunk size
- Monitor vector store size

**Import Errors**

- Ensure all dependencies are installed
- Check Python version compatibility
- Verify virtual environment activation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Format code
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Streamlit** for the web interface
- **FAISS** for efficient vector search
- **SentenceTransformers** for local embeddings
- **PyMuPDF** for PDF processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ali-mansha1351/AI_RAG_ResearchAssistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ali-mansha1351/AI_RAG_ResearchAssistant/discussions)

---

**Made with ❤️ for researchers and knowledge workers**

_Transform your documents into an intelligent research assistant!_
