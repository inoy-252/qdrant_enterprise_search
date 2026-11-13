"""
Enterprise Vector Search & Retrieval Engine
Stage 7: Native Hybrid Search & Reciprocal Rank Fusion (RRF) in Qdrant
Phase 1: Lexical Sparse Vector Encoding and Dual-Vector Schema Definition
"""

import math
import os
import zlib
from collections import Counter

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()


class BM25SparseEncoder:
    """
    In-memory sparse lexical encoder generating term-frequency vectors.
    Uses CRC32 deterministic hashing to map tokens to unsigned 32-bit sparse vector indices.
    """

    def __init__(self, max_dim: int = 2**31 - 1):
        self.max_dim = max_dim
        self.stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
            "of", "with", "is", "are", "was", "were", "it", "this", "that",
        }

    def tokenize(self, text: str) -> list[str]:
        """Extracts normalized alphanumeric tokens excluding common stop words."""
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        return [t for t in cleaned.split() if t not in self.stop_words and len(t) > 1]

    def encode(self, text: str) -> models.SparseVector:
        """
        Encodes input text into a Qdrant SparseVector with deterministic indices
        and log-scaled term frequencies (TF weighting).
        """
        tokens = self.tokenize(text)
        if not tokens:
            return models.SparseVector(indices=[0], values=[0.0])

        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for token, count in counts.items():
            token_hash = zlib.crc32(token.encode("utf-8")) % self.max_dim
            tf_weight = 1.0 + math.log(count)
            indices.append(int(token_hash))
            values.append(float(round(tf_weight, 4)))

        # Qdrant sparse vectors must be strictly sorted by index in ascending order
        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        return models.SparseVector(
            indices=[p[0] for p in sorted_pairs],
            values=[p[1] for p in sorted_pairs],
        )


class QdrantHybridEngine:
    """
    Production-grade hybrid search engine orchestrating dual-vector representation:
    1. Dense Vectors: 768-dimensional Gemini semantic embeddings via HNSW graph indexing.
    2. Sparse Vectors: BM25 lexical term-frequency vectors via Qdrant Sparse Inverted Index.
    """

    def __init__(
        self,
        collection_name: str = "enterprise_hybrid_catalog",
        db_path: str = "./qdrant_storage",
        vector_size: int = 768,
    ):
        self.collection_name = collection_name
        self.db_path = db_path
        self.vector_size = vector_size

        self.client = QdrantClient(path=self.db_path)
        self.sparse_encoder = BM25SparseEncoder()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
        self._genai_client = genai.Client(api_key=api_key)

        self._init_collection()

    def _init_collection(self):
        """Provisions a dual-vector collection with dense HNSW and sparse inverted indices."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
