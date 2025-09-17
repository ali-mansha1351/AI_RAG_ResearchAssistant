"""Application settings module."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # API key
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")

    # application
    app_name: str = Field(default="AI Research Assistant", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")

    # paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    vector_store_path: Path = Field(default_factory=lambda: Path("./data/vector_store"))
    documents_path: Path = Field(default_factory=lambda: Path("./data/processed"))
    logs_dir: Path = project_root / "logs"

    # model settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    use_local_embeddings: bool = Field(default=True, alias="USE_LOCAL_EMBEDDINGS")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    llm_model: str = Field(default="llama3-8b-8192", alias="LLM_MODEL")  # Groq model
    chunk_size: int = Field(default=1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=400, alias="CHUNK_OVERLAP")
    max_tokens: int = Field(default=2000, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    # API settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    
    # Additional common environment variables
    debug: bool = Field(default=False, alias="DEBUG")
    port: int = Field(default=8000, alias="PORT")  # Common for deployment platforms

    # logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default="./logs/app.log", alias="LOG_FILE")

    # vector store settings
    vector_store_type: str = Field(default="faiss", alias="VECTOR_STORE_TYPE")
    similarity_threshold: float = Field(default=0.3, alias="SIMILARITY_THRESHOLD")
    max_results: int = Field(default=15, alias="MAX_RESULTS")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields and ignore them
    }

    def __post_init__(self):
        """Create necessary directories."""
        self.data_dir.mkdir(exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.documents_path.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

settings = Settings()