import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class PineconeService:
    """Service to interact with Pinecone vector database.
    
    Includes an in-memory fallback when Pinecone credentials are not supplied,
    allowing full local development and testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key if api_key is not None else settings.PINECONE_API_KEY
        self.index_name = index_name if index_name is not None else settings.PINECONE_INDEX_NAME
        self.mock_mode = mock_mode
        self._client = None
        self._index = None
        self._local_docs_cache: Optional[List[Dict[str, Any]]] = None

        if not self.mock_mode and self.is_configured:
            self._init_pinecone()

    @property
    def is_configured(self) -> bool:
        """Return True if a valid Pinecone key and index name are configured."""
        return bool(
            self.api_key
            and self.api_key != "your_pinecone_api_key"
            and self.index_name
        )

    def _init_pinecone(self) -> None:
        """Initialize Pinecone client and index connection."""
        try:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.api_key)
            # Check if index exists or connect
            existing_indexes = [idx["name"] for idx in self._client.list_indexes()]
            if self.index_name in existing_indexes:
                self._index = self._client.Index(self.index_name)
                logger.info(f"Connected to Pinecone index: '{self.index_name}'")
            else:
                logger.warning(
                    f"Pinecone index '{self.index_name}' not found in {existing_indexes}. "
                    "Use index creation or setup script."
                )
        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {e}")
            self._client = None
            self._index = None

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Pinecone for the top_k most similar vectors.
        
        Falls back to in-memory cosine similarity search over local knowledge base
        if Pinecone is unconfigured or unreachable.
        """
        if self._index is not None and not self.mock_mode:
            try:
                response = self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                )
                results = []
                for match in response.get("matches", []):
                    metadata = match.get("metadata", {})
                    results.append(
                        {
                            "id": match.get("id"),
                            "score": float(match.get("score", 0.0)),
                            "title": metadata.get("title", "Untitled Reference"),
                            "category": metadata.get("category", "General"),
                            "section": metadata.get("section", ""),
                            "text": metadata.get("text", ""),
                        }
                    )
                return results
            except Exception as e:
                logger.warning(
                    f"Pinecone query failed ({e}); falling back to local semantic knowledge search."
                )

        # Fallback to local semantic search using cached knowledge docs
        return self._local_vector_search(query_vector, top_k)

    def upsert_records(
        self,
        records: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> int:
        """Upsert document vector records into Pinecone.
        
        Args:
            records: List of dicts with 'id', 'values' (vector), and 'metadata'.
        """
        if self._index is not None and not self.mock_mode:
            batch_size = 100
            total_upserted = 0
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self._index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
            logger.info(f"Upserted {total_upserted} records into Pinecone.")
            return total_upserted
        else:
            logger.info(
                f"Pinecone not connected. Recorded {len(records)} items in mock/local mode."
            )
            return len(records)

    def _load_local_docs(self) -> List[Dict[str, Any]]:
        """Loads the gardening documentation dataset for fallback search."""
        if self._local_docs_cache is None:
            data_file = settings.DATA_PATH
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    self._local_docs_cache = json.load(f)
            else:
                self._local_docs_cache = []
        return self._local_docs_cache

    def _local_vector_search(
        self, query_vector: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """In-memory cosine similarity search over local gardening docs."""
        from app.services.embedding_service import get_embedding_service

        docs = self._load_local_docs()
        if not docs:
            return []

        emb_service = get_embedding_service()
        texts = [doc["text"] for doc in docs]
        doc_embeddings = emb_service.get_embeddings(texts)

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scores = []
        for i, doc_emb in enumerate(doc_embeddings):
            d_vec = np.array(doc_emb, dtype=np.float32)
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 0:
                similarity = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
            else:
                similarity = 0.0
            scores.append((similarity, docs[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = scores[:top_k]

        results = []
        for sim, doc in top_matches:
            results.append(
                {
                    "id": doc.get("id"),
                    "score": sim,
                    "title": doc.get("title", "Gardening Reference"),
                    "category": doc.get("category", "General"),
                    "section": doc.get("section", ""),
                    "text": doc.get("text", ""),
                }
            )
        return results


_pinecone_service_instance: Optional[PineconeService] = None

def get_pinecone_service() -> PineconeService:
    """Singleton accessor for PineconeService."""
    global _pinecone_service_instance
    if _pinecone_service_instance is None:
        _pinecone_service_instance = PineconeService()
    return _pinecone_service_instance
