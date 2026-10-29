import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

class TestFastAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_serve_home_page(self):
        """Verify that GET / returns the Floramind HTML interface."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Floramind", response.text)
        self.assertIn("Google Gemini", response.text)
        self.assertIn("Pinecone", response.text)

    def test_health_endpoint(self):
        """Verify that GET /api/health returns valid system status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("gemini_model", data)
        self.assertIn("embedding_model", data)
        self.assertIn("pinecone_index", data)

    @patch("app.main.get_rag_service")
    def test_chat_endpoint_success(self, mock_get_rag):
        """Verify POST /api/chat calls RAG pipeline and returns structured response."""
        from app.models.schemas import ChatResponse, CitationSource

        mock_rag_instance = MagicMock()
        mock_rag_instance.answer_question.return_value = ChatResponse(
            answer="Here is how you start seeds.",
            citations=[
                CitationSource(
                    id="doc-seeds-01",
                    title="Seed Germination",
                    category="Seeds",
                    section="Germination",
                    snippet="Seeds require moisture, warmth, and oxygen.",
                    score=0.88,
                )
            ],
            model_used="gemini-2.5-flash",
            retrieval_status="grounded",
            is_grounded=True,
        )
        mock_get_rag.return_value = mock_rag_instance

        payload = {
            "message": "How do I start seeds?",
            "history": [],
            "top_k": 5,
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Here is how you start seeds.")
        self.assertTrue(data["is_grounded"])
        self.assertEqual(len(data["citations"]), 1)
        self.assertEqual(data["citations"][0]["id"], "doc-seeds-01")

    def test_chat_endpoint_empty_message_validation(self):
        """Verify that sending empty message string fails validation."""
        response = self.client.post("/api/chat", json={"message": ""})
        self.assertEqual(response.status_code, 422)  # Pydantic validation error


if __name__ == "__main__":
    unittest.main()
