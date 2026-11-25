import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hybrid_search_engine import BM25SparseEncoder, QdrantHybridEngine

SAMPLE_TEST_DOCS = [
    {
        "id": 1,
        "title": "Network Gateway Troubleshooting Guide",
        "content": "For packet drops on edge proxies, inspect firewall rule ERR-NET-9002. Restart interface eth0 if latency exceeds 200ms.",
        "metadata": {"department": "infrastructure", "tier": "p1"},
    },
    {
        "id": 2,
        "title": "Corporate Travel and Airfare Expense Policy",
        "content": "Employees may claim reimbursement for economy flights booked 14 days in advance. Meals capped at $75 per diem.",
        "metadata": {"department": "finance", "tier": "general"},
    },
    {
        "id": 3,
        "title": "Identity and Access Management SSO Protocol",
        "content": "Authentication failure SEC-AUTH-401 indicates expired SAML assertion. Reset OAuth2 tokens in enterprise portal.",
        "metadata": {"department": "security", "tier": "p0"},
    },
]


class TestBM25Encoder(unittest.TestCase):
    """Test suite for the BM25 lexical tokenizer."""

    def test_tokenization_removes_stopwords_and_punctuation(self):
        encoder = BM25SparseEncoder()
        text = "The firewall rule ERR-NET-9002 is active in the proxy!"

        tokens = encoder.tokenize(text)

        self.assertIn("firewall", tokens)
        self.assertIn("rule", tokens)
        self.assertIn("9002", tokens)

        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("in", tokens)

    def test_term_frequency_weighting(self):
        """Verify repeated words get higher log-scaled weight: 1.0 + ln(count)."""
        encoder = BM25SparseEncoder()
        text = "firewall firewall firewall"
        vector = encoder.encode(text)

        # There is only 1 unique word: 'firewall'
        self.assertEqual(len(vector.indices), 1)

        # Count is 3 -> 1.0 + ln(3) is approx 2.0986
        # So the weight must be greater than 1.5
        self.assertGreater(vector.values[0], 1.5)

    def test_empty_string_returns_safe_zero_vector(self):
        """Verify edge case where input contains only whitespace or stopwords."""
        encoder = BM25SparseEncoder()

        # Test empty whitespace
        empty_vector = encoder.encode("   ")
        self.assertEqual(empty_vector.indices, [0])
        self.assertEqual(empty_vector.values, [0.0])

        # Test stopword-only string
        stopword_vector = encoder.encode("the is at were")
        self.assertEqual(stopword_vector.indices, [0])
        self.assertEqual(stopword_vector.values, [0.0])

    def test_deterministic_indices(self):
        """Verify that identical words always map to the exact same sparse index."""
        encoder = BM25SparseEncoder()
        vec1 = encoder.encode("kubernetes cluster")
        vec2 = encoder.encode("kubernetes cluster")

        # Both runs must produce identical vector indices and values
        self.assertEqual(vec1.indices, vec2.indices)
        self.assertEqual(vec1.values, vec2.values)


class TestQdrantHybridEngine(unittest.TestCase):
    """Integration tests for Qdrant dual-vector indexing and hybrid search."""

    @classmethod
    def setUpClass(cls):
        # 1. Create a temporary folder on disk
        cls.temp_dir = tempfile.mkdtemp(prefix="test_qdrant_")

        # 2. Set dummy API key
        os.environ["GEMINI_API_KEY"] = "mock-key-for-offline-tests"

        # 3. Setup our stunt double (Mock) for Google GenAI
        mock_embedding = [0.05] * 768
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=mock_embedding)]

        cls.mock_patch = patch("hybrid_search_engine.genai.Client")
        cls.mock_client_class = cls.mock_patch.start()
        cls.mock_instance = MagicMock()
        cls.mock_instance.models.embed_content.return_value = mock_response
        cls.mock_client_class.return_value = cls.mock_instance

        # 4. Initialize engine ONCE for all tests in this class
        cls.engine = QdrantHybridEngine(
            collection_name="test_catalog",
            db_path=cls.temp_dir,
            vector_size=768,
        )

        # 5. Index the test documents once
        cls.engine.index_batch(SAMPLE_TEST_DOCS)

    @classmethod
    def tearDownClass(cls):
        # 1. Cleanly close the Qdrant client to release the disk lock
        if hasattr(cls, "engine") and hasattr(cls.engine, "client"):
            cls.engine.client.close()

        # 2. Stop the mock interceptor
        cls.mock_patch.stop()

        # 3. Delete the temporary folder
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_telemetry(self):
        """Verify collection point count and vector dimension in telemetry."""
        telemetry = self.engine.get_telemetry()
        # Notice we check the exact keys returned by get_telemetry()
        self.assertEqual(telemetry["collection"], "test_catalog")
        self.assertEqual(telemetry["total_points"], 3)
        self.assertEqual(telemetry["dense_vector_dim"], 768)

    def test_sparse_lexical_search_finds_exact_code(self):
        """Verify BM25 sparse search retrieves Document 1 for exact code 'ERR-NET-9002'."""
        hits = self.engine.sparse_search("ERR-NET-9002", limit=1)

        self.assertGreater(len(hits), 0, "Sparse search should return at least 1 hit.")
        self.assertEqual(
            hits[0]["id"], 1, "Top hit for ERR-NET-9002 must be Document 1."
        )
        self.assertIn("firewall rule ERR-NET-9002", hits[0]["content"])

    def test_hybrid_search_returns_payload_and_rrf_score(self):
        """Verify Hybrid search merges dense and sparse results with valid score and payload."""
        hits = self.engine.hybrid_search(
            "Corporate Travel flight reimbursement", limit=1
        )

        self.assertGreater(len(hits), 0, "Hybrid search should return at least 1 hit.")
        top_hit = hits[0]
        self.assertIn("id", top_hit)
        self.assertIn("title", top_hit)
        self.assertIn("score", top_hit)
        self.assertIn("content", top_hit)
        self.assertIn("metadata", top_hit)
        self.assertGreater(top_hit["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
