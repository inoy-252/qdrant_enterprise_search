import os 

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models 

load_dotenv()

class QdrantSemanticCache:
    """
    Semantic caching and vector retrieval engine using Qdrant.
    Uses Google Gemini embeddings and HNSW graph indexing for sub-millisecond search.
    """
    def __init__(
        self,
        collection_name: str = "semantic_cache",
        db_path: str = "./qdrant_storage",
        threshold: float = 0.90,
        vector_size: int  = 768,
    ):
        self.collection_name = collection_name
        self.db_path = db_path
        self.threshold = threshold
        self.vector_size = vector_size 
        self.client = QdrantClient(path=self.db_path)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
        self._genai_client = genai.Client(api_key=api_key)
        
        self._init_db()

    def _init_db(self):
        """Ensures the Qdrant collection exists with HNSW cosine indexing."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Generates a 768-dimensional dense vector using Gemini."""
        response = self._genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values