import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
from datetime import datetime as dt

import polars as pl
import ijson

from axis_python.functions.cosine_similarity import cosine_similarity

logger = logging.getLogger(__name__)



class VectorRegistry:
    """Manages vector storage and retrieval for a collection."""

    def __init__(self, path: str):
        self.path = path
        self.collection_name: str = "main"
        self.vectors: List[List[float]] = []
        self.created_datetime: List[dt] = []
        self.origin_datetime: Optional[dt] = None

    def lazy_load(self, collection: str = "main") -> None:
        """Load only vectors from the specified collection, keeping payloads for lazy loading."""
        self.collection_name = collection
        if not os.path.exists(self.path):
            logger.info(f"Database file {self.path} does not exist yet, starting with empty data")
            return

        try:
            with open(self.path, "rb") as f:
                # Stream only vectors from target collection
                vector_count = 0
                for vector in ijson.items(f, f"{collection}.vectors.item"):
                    self.vectors.append([float(v) for v in vector])
                    vector_count += 1
                
                if vector_count == 0:
                    logger.info(f"Collection '{collection}' not found or empty, starting with empty data")
                    return
                
                logger.info(f"Loaded {vector_count} vectors from {self.path}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.path}: {e}")
            raise

    def get_payload_at_index(self, payload_index: int) -> Dict[str, Any]:
        """Efficiently retrieve a specific payload by index without loading all payloads."""
        if payload_index < 0 or payload_index >= len(self.vectors):
            raise IndexError(f"Payload index {payload_index} out of range")
        
        try:
            with open(self.path, "rb") as f:
                for index, payload in enumerate(ijson.items(f, f"{self.collection_name}.payloads.item")):
                    if index == payload_index:
                        return payload
            
            raise IndexError(f"Payload at index {payload_index} not found")
        except Exception as e:
            logger.error(f"Failed to retrieve payload at index {index}: {e}")
            raise

    def save(self) -> None:        
        payloads_to_save = getattr(self, '_insertion_payloads', [])
        

        if payloads_to_save:
            existing_payloads = []
            try:
                with open(self.path, "rb") as f:
                    for payload in ijson.items(f, f"{self.collection_name}.payloads.item"):
                        existing_payloads.append(payload)
            except:
                pass
            
            all_payloads = existing_payloads + payloads_to_save
            self._insertion_payloads = [] 
        else:
            all_payloads = []
            try:
                with open(self.path, "rb") as f:
                    for payload in ijson.items(f, f"{self.collection_name}.payloads.item"):
                        all_payloads.append(payload)
            except:
                pass
        
        collection_data = {
            "vectors": self.vectors,
            "payloads": all_payloads,
            "created_datetime": [d.isoformat() if isinstance(d, dt) else d for d in self.created_datetime],
            "origin_datetime": self.origin_datetime.isoformat() if self.origin_datetime else None,
        }
        
        data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read existing file, overwriting: {e}")
        

        data[self.collection_name] = collection_data
        
        try:
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved collection '{self.collection_name}' to {self.path}")
        except Exception as e:
            logger.error(f"Failed to save {self.path}: {e}")
            raise

    @property
    def payload_count(self) -> int:
        """Return the number of payloads (same as vector count)."""
        return len(self.vectors)



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
        """Switch to a different collection, saving current state first."""
        if self._vector_registry is not None:
            self._vector_registry.save()
        
        self._vector_registry = VectorRegistry(self.path)
        self._vector_registry.lazy_load(collection)
        logger.info(f"Switched to collection '{collection}'")

    def insert(self, text: str, payload: Dict[str, Any]) -> None:
        """Embed text and store with payload."""
        vector = self.embedder.encode(text).tolist()
        payload["text"] = text
        
        self.vector_registry.vectors.append(vector)
        
        # For insertion, we need to temporarily store payloads to save
        if not hasattr(self.vector_registry, '_insertion_payloads'):
            self.vector_registry._insertion_payloads = []
        self.vector_registry._insertion_payloads.append(payload)
        
        self.vector_registry.created_datetime.append(dt.now())
        
        if self.vector_registry.origin_datetime is None:
            self.vector_registry.origin_datetime = dt.now()
        
        self.vector_registry.save()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search using cosine similarity, retrieving payloads on-demand."""
        if not self.vector_registry.vectors:
            return []
        
        q_vec = self.embedder.encode(query)
        scores = [
            cosine_similarity(q_vec, np.array(v))
            for v in self.vector_registry.vectors
        ]
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

