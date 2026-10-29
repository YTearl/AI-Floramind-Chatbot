import unittest
from unittest.mock import MagicMock
from app.services.rag_service import RAGService
from app.models.schemas import ChatMessage

class TestRAGService(unittest.TestCase):
    """Test suite for the RAG orchestration pipeline."""

    def setUp(self):
        self.mock_embedding = MagicMock()
        self.mock_embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.mock_embedding.get_embedding.return_value = [0.1] * 384

        self.mock_pinecone = MagicMock()
        self.mock_gemini = MagicMock()
        self.mock_gemini.model_name = "gemini-2.5-flash"

        self.rag_service = RAGService(
            embedding_service=self.mock_embedding,
            pinecone_service=self.mock_pinecone,
            gemini_service=self.mock_gemini,
            relevance_threshold=0.45,
        )

    def test_rag_flow_grounded_answer(self):
        """Test RAG flow when Pinecone returns highly relevant documents."""
        self.mock_pinecone.query.return_value = [
            {
                "id": "doc-compost-01",
                "score": 0.82,
                "title": "Home Composting",
                "category": "Soil & Compost",
                "section": "Basics",
                "text": "A balanced compost pile requires 2 to 3 parts brown to 1 part green.",
            },
            {
                "id": "doc-soil-01",
                "score": 0.65,
                "title": "Soil Preparation",
                "category": "Soil",
                "section": "Fundamentals",
                "text": "Healthy soil is the foundation.",
            },
        ]

        self.mock_gemini.generate_response.return_value = {
            "answer": "Composting needs 2 to 3 parts brown to 1 part green according to Home Composting.",
            "model_used": "gemini-2.5-flash",
            "method": "interactions_api",
        }

        response = self.rag_service.answer_question("How do I make compost?")

        self.assertTrue(response.is_grounded)
        self.assertEqual(response.retrieval_status, "grounded")
        self.assertEqual(len(response.citations), 2)
        self.assertEqual(response.citations[0].title, "Home Composting")
        self.assertEqual(response.citations[0].score, 0.82)
        self.mock_embedding.get_embedding.assert_called_once_with("How do I make compost?")
        self.mock_pinecone.query.assert_called_once()
        self.mock_gemini.generate_response.assert_called_once()

    def test_rag_flow_insufficient_relevance(self):
        """Test RAG flow when Pinecone returns low-scoring documents below threshold."""
        self.mock_pinecone.query.return_value = [
            {
                "id": "doc-unrelated",
                "score": 0.22,  # Well below 0.45 threshold
                "title": "Unrelated Topic",
                "category": "General",
                "section": "",
                "text": "Completely unrelated content.",
            }
        ]

        self.mock_gemini.generate_response.return_value = {
            "answer": "Our knowledge base has no specific documents on building rockets, but here is safe general info.",
            "model_used": "gemini-2.5-flash",
            "method": "interactions_api",
        }

        response = self.rag_service.answer_question("How do I build a rocket engine?")

        self.assertFalse(response.is_grounded)
        self.assertEqual(response.retrieval_status, "insufficient")
        # No citations should be displayed when relevance threshold fails
        self.assertEqual(len(response.citations), 0)

    def test_rag_flow_empty_retrieval(self):
        """Test RAG flow when Pinecone returns an empty list."""
        self.mock_pinecone.query.return_value = []
        self.mock_gemini.generate_response.return_value = {
            "answer": "No relevant documents found.",
            "model_used": "gemini-2.5-flash",
            "method": "mock_generator",
        }

        response = self.rag_service.answer_question("Obscure question?")
        self.assertFalse(response.is_grounded)
        self.assertEqual(response.retrieval_status, "insufficient")
        self.assertEqual(len(response.citations), 0)

    def test_empty_query_validation(self):
        """Test that empty queries are gracefully handled."""
        response = self.rag_service.answer_question("   ")
        self.assertFalse(response.is_grounded)
        self.assertIn("enter a gardening question", response.answer)


if __name__ == "__main__":
    unittest.main()
