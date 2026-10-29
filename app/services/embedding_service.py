import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service to generate dense vector embeddings using Hugging Face Sentence Transformers."""

    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.HF_EMBEDDING_MODEL
        self._load_model()

    def _load_model(self) -> None:
        """Loads the sentence-transformers model lazily."""
        if self._model is None:
            logger.info(f"Loading Hugging Face embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                raise

    def get_embedding(self, text: str) -> List[float]:
        """Convert a single text query into an embedding vector.
        
        Args:
            text: Query string or document text.
            
        Returns:
            List of floats representing the embedding.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        
        if self._model is None:
            self._load_model()

        embedding = self._model.encode(text.strip(), convert_to_numpy=True)
        return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Convert a batch of texts into embeddings.
        
        Args:
            texts: List of strings.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        
        if self._model is None:
            self._load_model()

        clean_texts = [t.strip() for t in texts]
        embeddings = self._model.encode(clean_texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


_embedding_service_instance: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    """Singleton accessor for EmbeddingService."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
