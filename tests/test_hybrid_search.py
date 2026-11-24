import unittest

from hybrid_search_engine import BM25SparseEncoder


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
