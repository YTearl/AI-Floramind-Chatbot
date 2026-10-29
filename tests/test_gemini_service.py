import unittest
from unittest.mock import MagicMock, patch
from app.services.gemini_service import GeminiService, FLORAMIND_SYSTEM_PROMPT
from app.models.schemas import ChatMessage

class TestGeminiService(unittest.TestCase):
    """Test suite for GeminiService ensuring robust error handling and mock support."""

    def setUp(self):
        self.service = GeminiService(mock_mode=True, model_name="gemini-2.5-flash")

    def test_mock_generation_grounded(self):
        """Verify mock response generation when relevant context docs are present."""
        context = [
            {
                "id": "doc-soil-01",
                "title": "Soil Preparation & Health",
                "category": "Soil & Compost",
                "section": "Fundamentals",
                "text": "Healthy soil is the foundation of every thriving garden.",
                "score": 0.88,
            }
        ]
        res = self.service.generate_response(
            query="How do I prepare soil?",
            context_docs=context,
            is_relevant=True,
        )
        self.assertIn("Soil Preparation & Health", res["answer"])
        self.assertIn("Floramind", res["answer"])
        self.assertEqual(res["method"], "mock_generator")

    def test_mock_generation_insufficient_relevance(self):
        """Verify limitation statement when relevance is low or references are empty."""
        res = self.service.generate_response(
            query="How do I build an airplane?",
            context_docs=[],
            is_relevant=False,
        )
        self.assertIn("don't currently have detailed reference documents", res["answer"])

    def test_prompt_construction(self):
        """Verify prompt includes grounding instructions and system rules."""
        context = [
            {
                "title": "Composting",
                "category": "Compost",
                "section": "Basics",
                "text": "2 to 3 parts brown to 1 part green.",
            }
        ]
        prompt = self.service.build_prompt("Compost tips", context, is_relevant=True)
        self.assertIn("Composting", prompt)
        self.assertIn("HIGHLY RELEVANT", prompt)
        self.assertIn("Compost tips", prompt)

    def test_error_handling_invalid_key(self):
        """Verify helpful error translation for invalid API keys."""
        res = self.service._handle_gemini_error(
            Exception("API_KEY_INVALID: 401 Unauthorized"),
            query="test",
            context_docs=[],
            is_relevant=False,
        )
        self.assertIn("invalid or unauthenticated", res["answer"])
        self.assertEqual(res["method"], "error_fallback")

    def test_error_handling_quota_exceeded(self):
        """Verify user-friendly guidance on quota exhaustion."""
        res = self.service._handle_gemini_error(
            Exception("RESOURCE_EXHAUSTED: 429 Quota exceeded"),
            query="test",
            context_docs=[],
            is_relevant=False,
        )
        self.assertIn("quota or rate limit has been temporarily exceeded", res["answer"])

    def test_error_handling_model_not_found(self):
        """Verify handling when model identifier is unavailable or mistyped."""
        res = self.service._handle_gemini_error(
            Exception("404 Model not found: gemini-invalid"),
            query="test",
            context_docs=[],
            is_relevant=False,
        )
        self.assertIn("could not be reached", res["answer"])

    def test_interactions_api_fallback_to_generate_content(self):
        """Verify fallback from Interactions API to client.models.generate_content."""
        service = GeminiService(api_key="valid-dummy-key", model_name="gemini-2.5-flash")
        
        # Mock client with failing interactions and working generate_content
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = Exception("Interactions API preview unavailable")
        
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini generate_content! Composting requires 2 parts brown."
        mock_client.models.generate_content.return_value = mock_response
        
        service._client = mock_client
        
        res = service.generate_response(
            query="Composting guide",
            context_docs=[{"title": "Composting", "category": "Soil", "text": "2 parts brown"}],
            is_relevant=True,
        )
        
        self.assertEqual(res["method"], "models_generate_content")
        self.assertIn("Hello from Gemini", res["answer"])
        mock_client.interactions.create.assert_called_once()
        mock_client.models.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
