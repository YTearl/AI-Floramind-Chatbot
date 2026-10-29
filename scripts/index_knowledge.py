import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.services.embedding_service import get_embedding_service
from app.services.pinecone_service import get_pinecone_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_knowledge")

def index_gardening_knowledge(data_file: Path = settings.DATA_PATH) -> int:
    """Reads gardening docs, embeds text using Hugging Face, and upserts into Pinecone."""
    if not data_file.exists():
        logger.error(f"Knowledge data file not found at: {data_file}")
        return 0

    with open(data_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    logger.info(f"Loaded {len(docs)} gardening documents from {data_file}")

    # Initialize embedding service
    emb_service = get_embedding_service()
    texts = [doc["text"] for doc in docs]
    logger.info(f"Generating embeddings using {emb_service.model_name}...")
    embeddings = emb_service.get_embeddings(texts)

    # Check Pinecone
    pinecone_svc = get_pinecone_service()
    if not pinecone_svc.is_configured:
        logger.warning(
            "Pinecone is not fully configured (check PINECONE_API_KEY in .env). "
            "Documents will be ready for local fallback search."
        )
        return len(docs)

    # If Pinecone is configured, create index if needed and upsert
    try:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        existing = [idx["name"] for idx in pc.list_indexes()]
        
        dim = len(embeddings[0]) if embeddings else 384

        if settings.PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index '{settings.PINECONE_INDEX_NAME}' (dim={dim})...")
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Created index '{settings.PINECONE_INDEX_NAME}'.")

        records = []
        for doc, emb in zip(docs, embeddings):
            records.append(
                {
                    "id": doc["id"],
                    "values": emb,
                    "metadata": {
                        "title": doc.get("title", ""),
                        "category": doc.get("category", "General"),
                        "section": doc.get("section", ""),
                        "text": doc.get("text", ""),
                    },
                }
            )

        upserted_count = pinecone_svc.upsert_records(records)
        logger.info(f"Successfully indexed {upserted_count} documents in Pinecone!")
        return upserted_count

    except Exception as e:
        logger.error(f"Failed to upsert to Pinecone: {e}")
        return 0


if __name__ == "__main__":
    count = index_gardening_knowledge()
    print(f"Indexing complete. Processed {count} documents.")
