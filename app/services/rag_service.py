import logging
from typing import List, Optional
from app.config import settings
from app.models.schemas import ChatMessage, ChatResponse, CitationSource
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.pinecone_service import PineconeService, get_pinecone_service
from app.services.gemini_service import GeminiService, get_gemini_service

logger = logging.getLogger(__name__)

class RAGService:
    """Coordinates the Retrieval-Augmented Generation (RAG) pipeline for Floramind.
    
    1. Hugging Face: Encodes user query to dense vector embedding.
    2. Pinecone: Retrieves top 5 most relevant gardening document chunks.
    3. Relevance Validation: Evaluates similarity scores against relevance threshold.
    4. Google Gemini API: Generates grounded conversational answer as Floramind mentor.
    5. Attribution: Formats verified citation sources for frontend display.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        pinecone_service: Optional[PineconeService] = None,
        gemini_service: Optional[GeminiService] = None,
        relevance_threshold: Optional[float] = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.pinecone_service = pinecone_service or get_pinecone_service()
        self.gemini_service = gemini_service or get_gemini_service()
        self.relevance_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else settings.RELEVANCE_THRESHOLD
        )

    def answer_question(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None,
        top_k: int = 5,
    ) -> ChatResponse:
        """Process user question through full RAG lifecycle."""
        clean_query = query.strip()
        if not clean_query:
            return ChatResponse(
                answer="Please enter a gardening question so I can assist you!",
                citations=[],
                model_used=self.gemini_service.model_name,
                retrieval_status="insufficient",
                is_grounded=False,
            )

        # Step 1: Generate query embedding via Hugging Face sentence-transformers
        logger.info(f"Generating query embedding with model: {self.embedding_service.model_name}")
        try:
            query_embedding = self.embedding_service.get_embedding(clean_query)
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return ChatResponse(
                answer=(
                    "🌿 **Notice from Floramind:** I encountered a temporary issue generating embeddings "
                    "for your question. Please ensure the Hugging Face sentence-transformers model is loaded."
                ),
                citations=[],
                model_used=self.gemini_service.model_name,
                retrieval_status="error",
                is_grounded=False,
            )

        # Step 2: Search Pinecone for top 5 most relevant chunks
        logger.info(f"Querying vector database for top {top_k} gardening references...")
        try:
            retrieved_docs = self.pinecone_service.query(
                query_vector=query_embedding, top_k=top_k
            )
        except Exception as e:
            logger.error(f"Pinecone retrieval error: {e}")
            retrieved_docs = []

        # Step 3: Check whether retrieved documents are sufficiently relevant
        top_score = retrieved_docs[0]["score"] if retrieved_docs else 0.0
        is_relevant = bool(retrieved_docs and top_score >= self.relevance_threshold)

        if not retrieved_docs:
            retrieval_status = "insufficient"
            relevant_docs = []
        elif is_relevant:
            retrieval_status = "grounded"
            # Filter documents that meet a reasonable fraction of top score
            relevant_docs = [
                d for d in retrieved_docs if d["score"] >= (self.relevance_threshold * 0.8)
            ]
        else:
            retrieval_status = "insufficient"
            relevant_docs = []

        logger.info(
            f"Retrieval status: {retrieval_status} (top_score={top_score:.3f}, "
            f"threshold={self.relevance_threshold})"
        )

        # Step 4 & 5: Pass context, question, and history to Gemini
        gemini_result = self.gemini_service.generate_response(
            query=clean_query,
            context_docs=relevant_docs if is_relevant else retrieved_docs[:2],
            history=history,
            is_relevant=is_relevant,
        )

        # Step 6 & 7: Prepare verified citation metadata
        citations: List[CitationSource] = []
        if is_relevant and relevant_docs:
            for doc in relevant_docs:
                text_snippet = doc.get("text", "")
                if len(text_snippet) > 220:
                    text_snippet = text_snippet[:217] + "..."
                citations.append(
                    CitationSource(
                        id=str(doc.get("id")),
                        title=str(doc.get("title", "Gardening Reference")),
                        category=doc.get("category"),
                        section=doc.get("section"),
                        snippet=text_snippet,
                        score=round(float(doc.get("score", 0.0)), 3),
                    )
                )

        return ChatResponse(
            answer=gemini_result.get("answer", ""),
            citations=citations,
            model_used=gemini_result.get("model_used", self.gemini_service.model_name),
            retrieval_status=retrieval_status,
            is_grounded=is_relevant and len(citations) > 0,
        )


_rag_service_instance: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """Singleton accessor for RAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
