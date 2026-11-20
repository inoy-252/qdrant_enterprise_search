import math
import os
import time
import uuid
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
            "a",
            "an",
            "the",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "is",
            "are",
            "was",
            "were",
            "it",
            "this",
            "that",
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
        indices = []
        values = []

        for token, count in counts.items():
            token_hash = zlib.crc32(token.encode("utf-8")) % self.max_dim
            tf_weight = 1.0 + math.log(count)
            indices.append(int(token_hash))
            values.append(float(round(tf_weight, 4)))

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

    def _get_dense_embedding(self, text: str) -> list[float]:
        """Generates 768-dimensional dense vector using Gemini with MRL truncation."""
        response = self._genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                "output_dimensionality": self.vector_size,
                "task_type": "SEMANTIC_SIMILARITY",
            },
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("Gemini API failed to return dense embedding vectors")
        return response.embeddings[0].values

    def index_batch(self, documents: list[dict]) -> list[str]:
        """
        Batch indexes multiple documents with dual-vector representations
        in a single atomic upsert operation.
        Each document dict expects:
            - 'title': str
            - 'content': str
            - 'id': str (optional)
            - 'metadata': dict (optional)
        """
        if not documents:
            return []
        points = []
        point_ids = []

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            point_id = doc.get("id") or str(uuid.uuid4())
            full_text = f"{title}\n{content}".strip()

            dense_vector = self._get_dense_embedding(full_text)
            sparse_vector = self.sparse_encoder.encode(full_text)

            payload = {
                "title": title,
                "content": content,
                "indexed_at": time.time(),
            }
            metadata = doc.get("metadata")
            if metadata and isinstance(metadata, dict):
                for k, v in metadata.items():
                    payload[k] = v.strip().lower() if isinstance(v, str) else v
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector,
                    },
                    payload=payload,
                )
            )
            point_ids.append(point_id)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return point_ids

    def index_document(
        self,
        title: str,
        content: str,
        doc_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Indexes a single document by delegating to index_batch."""
        doc: dict = {"title": title, "content": content}
        if doc_id:
            doc["id"] = doc_id
        if metadata:
            doc["metadata"] = metadata
        return self.index_batch([doc])[0]

    def _build_filter(self, filter_metadata: dict | None) -> models.Filter | None:
        """Constructs a Qdrant Filter from a key-value dictionary."""
        if not filter_metadata:
            return None
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=k,
                match=models.MatchValue(
                    value=v.strip().lower() if isinstance(v, str) else v
                ),
            )
            for k, v in filter_metadata.items()
        ]
        return models.Filter(must=conditions)

    def _format_results(self, points: list) -> list[dict]:
        """Formats raw Qdrant ScoredPoints into clean dictionary outputs."""
        results = []
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "id": point.id,
                    "score": round(point.score, 4),
                    "title": payload.get("title", ""),
                    "content": payload.get("content", ""),
                    "metadata": {
                        k: v
                        for k, v in payload.items()
                        if k not in ["title", "content", "indexed_at"]
                    },
                }
            )
        return results

    def hybrid_search(
        self, query: str, limit: int = 5, filter_metadata: dict | None = None
    ) -> list[dict]:
        """
        Executes native hybrid search inside Qdrant:
        - Prefetches dense candidates via HNSW semantic search.
        - Prefetches sparse candidates via inverted lexical index.
        - Fuses candidate rankings using Reciprocal Rank Fusion (RRF).
        """

        query_dense = self._get_dense_embedding(query)
        query_sparse = self.sparse_encoder.encode(query)
        query_filter = self._build_filter(filter_metadata)

        prefetch_requests = [
            models.Prefetch(
                query=query_dense,
                using="dense",
                filter=query_filter,
                limit=limit * 2,
            ),
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                filter=query_filter,
                limit=limit * 2,
            ),
        ]
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_requests,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return self._format_results(response.points)

    def dense_search(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Executes pure dense semantic search using 768-D Gemini embeddings."""
        query_dense = self._get_dense_embedding(query)
        query_filter = self._build_filter(filter_metadata)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            query_filter=query_filter,
            limit=limit,
        )
        return self._format_results(response.points)

    def sparse_search(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Executes pure sparse lexical search using BM25 token frequencies."""
        query_sparse = self.sparse_encoder.encode(query)
        query_filter = self._build_filter(filter_metadata)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,
            using="sparse",
            query_filter=query_filter,
            limit=limit,
        )
        return self._format_results(response.points)

    def get_telemetry(self) -> dict:
        """Retrieves collection statistics, point counts, and index health."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "collection": self.collection_name,
            "total_points": info.points_count,
            "status": info.status.name.lower() if info.status else "unknown",
            "dense_vector_dim": self.vector_size,
            "sparse_vectors_enabled": True,
            "fusion_algorithm": "Reciprocal Rank Fusion (RRF)",
        }
