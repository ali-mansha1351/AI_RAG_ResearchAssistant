# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --timeout=1000 --retries=5 \
    sentence-transformers langchain-community 

# Copy the entire application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create necessary directories
RUN mkdir -p data/raw data/processed data/vector_store logs

# Expose the port Streamlit runs on
EXPOSE 8501

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
