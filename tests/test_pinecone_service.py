import unittest
from unittest.mock import MagicMock
from app.services.pinecone_service import PineconeService

class TestPineconeService(unittest.TestCase):
    """Test suite for Pinecone vector search and mock/fallback operations."""

    def test_mock_query_fallback(self):
        """Verify that local vector search provides results when Pinecone is unconfigured."""
        service = PineconeService(mock_mode=True)
        # Dummy 384-dimensional vector
        dummy_vector = [0.05] * 384
        results = service.query(query_vector=dummy_vector, top_k=3)
        
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("title", results[0])
        self.assertIn("score", results[0])
        self.assertIn("text", results[0])

    def test_pinecone_index_query_success(self):
        """Verify query formatting when Pinecone index returns matches."""
        service = PineconeService(mock_mode=False)
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc-tropical-01",
                    "score": 0.892,
                    "metadata": {
                        "title": "Tropical Gardening",
                        "category": "Tropical",
                        "section": "Climate",
                        "text": "Tropical climates feature warm temperatures year-round.",
                    },
                }
            ]
        }
        service._index = mock_index

        results = service.query(query_vector=[0.1] * 384, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc-tropical-01")
        self.assertEqual(results[0]["title"], "Tropical Gardening")
        self.assertAlmostEqual(results[0]["score"], 0.892)

    def test_upsert_records_mock(self):
        """Verify upserting records returns record count without errors in mock mode."""
        service = PineconeService(mock_mode=True)
        records = [
            {"id": "test-1", "values": [0.1] * 384, "metadata": {"title": "Test 1"}},
            {"id": "test-2", "values": [0.2] * 384, "metadata": {"title": "Test 2"}},
        ]
        upserted = service.upsert_records(records)
        self.assertEqual(upserted, 2)


if __name__ == "__main__":
    unittest.main()
