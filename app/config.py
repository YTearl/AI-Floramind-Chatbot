import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory for the Floramind project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    """Application configuration loaded securely from environment variables."""

    # Google Gemini Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    # Hugging Face Embedding Settings
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()

    # Pinecone Vector Database Settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "").strip()
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "floramind-gardening").strip()

    # RAG Retrieval Settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))

    # Server settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_PATH: Path = BASE_DIR / "data" / "gardening_docs.json"

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a Gemini API key is provided."""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY != "your_gemini_api_key")

    @property
    def is_pinecone_configured(self) -> bool:
        """Check if Pinecone API key and index are provided."""
        return bool(
            self.PINECONE_API_KEY
            and self.PINECONE_API_KEY != "your_pinecone_api_key"
            and self.PINECONE_INDEX_NAME
        )


settings = Settings()
