import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os

from axis_python.vector_registry import VectorRegistry
from axis_python.functions.cosine_similarity import cosine_similarity

logger = logging.getLogger(__name__)


class aXisDB:
    """Vector database with semantic search capabilities."""

    def __init__(self, path: str = "axis.db"):
        self.path = path
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(model_path, local_files_only=True)
        self._vector_registry: Optional[VectorRegistry] = None

    @property
    def vector_registry(self) -> VectorRegistry:
        """Lazy-load vector registry on first access."""
        if self._vector_registry is None:
            self._vector_registry = VectorRegistry(self.path)
            self._vector_registry.lazy_load()
        return self._vector_registry

    def switch_collection(self, collection: str = "main") -> None:
        """
        Description:
            Switch to a different collection, saving current state first.
        Args:
            collection: str
                Name of the collection to switch to (default: "main")
        """
        if self._vector_registry is not None:
            self._vector_registry.save()
        
        self._vector_registry = VectorRegistry(self.path)
        self._vector_registry.lazy_load(collection)
        logger.info(f"Switched to collection '{collection}'")

    def insert(self, text: str, payload: Dict[str, Any]) -> None:
        """
        Description:
            Embed text and store with payload.
        """
        vector = self.embedder.encode(text).astype(np.float32)
        payload["text"] = text
        
        # Append vector to numpy array
        if len(self.vector_registry.vectors) == 0:
            self.vector_registry.vectors = vector.reshape(1, -1)
        else:
            self.vector_registry.vectors = np.vstack([self.vector_registry.vectors, vector])
        
        # Store insertion (payload only; vector already in self.vectors)
        if not hasattr(self.vector_registry, '_insertion_vector'):
            self.vector_registry._insertion_vector = []
        self.vector_registry._insertion_vector.append(payload)
        
        self.vector_registry.save()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Description:
            Semantic search using cosine similarity, retrieving payloads on-demand.
        """
        if len(self.vector_registry.vectors) == 0:
            return []
        
        q_vec = self.embedder.encode(query).astype(np.float32)
        scores = np.array([
            cosine_similarity(q_vec, v)
            for v in self.vector_registry.vectors
        ])
        top_idx = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for i in top_idx:
            payload = self.vector_registry.get_payload_at_index(i)
            result = {
                "score": round(float(scores[i]), 4),
                "text": payload.get("text", ""),
            }
            result.update({k: v for k, v in payload.items() if k != "text"})
            results.append(result)
        
        return results

