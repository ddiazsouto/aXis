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
        self.payloads: List[Dict[str, Any]] = []
        self.created_datetime: List[dt] = []
        self.origin_datetime: Optional[dt] = None

    def load(self, collection: str = "main") -> None:
        """Load only the specified collection from disk, ignoring others."""
        self.collection_name = collection
        if not os.path.exists(self.path):
            logger.info(f"Database file {self.path} does not exist yet, starting with empty data")
            return

        try:
            with open(self.path, "rb") as f:
                # Use ijson to stream only the target collection
                parser = ijson.kvitems(f, "")
                for key, value in parser:
                    if key == collection:
                        self._parse_collection_data(value)
                        return
                
                # Collection not found
                logger.info(f"Collection '{collection}' not found, starting with empty data")
        except Exception as e:
            logger.error(f"Failed to load {self.path}: {e}")
            raise

    def _parse_collection_data(self, collection_data: Dict[str, Any]) -> None:
        """Parse collection data from JSON, converting datetimes and numbers as needed."""
        # Convert Decimal objects from ijson to floats
        vectors_data = collection_data.get("vectors", [])
        self.vectors = [
            [float(v) for v in vector] if isinstance(vector, list) else vector 
            for vector in vectors_data
        ]
        
        self.payloads = collection_data.get("payloads", [])
        
        created_dt_data = collection_data.get("created_datetime", [])
        self.created_datetime = [dt.fromisoformat(d) if isinstance(d, str) else d for d in created_dt_data]
        
        origin_dt = collection_data.get("origin_datetime")
        self.origin_datetime = dt.fromisoformat(origin_dt) if origin_dt else None
        
        logger.info(f"Loaded {len(self.vectors)} vectors from {self.path}")

    def save(self) -> None:        
        collection_data = {
            "vectors": self.vectors,
            "payloads": self.payloads,
            "created_datetime": [dt.isoformat() if isinstance(dt, dt.__class__) else dt for dt in self.created_datetime],
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

    def get_collection_dataframe(self, start_datetime: Optional[dt] = None) -> pl.DataFrame:
        filter_datetime = start_datetime or self.origin_datetime
        
        df = pl.DataFrame({
            "payload": self.payloads,
            "created_datetime": self.created_datetime,
        })
        
        if filter_datetime:
            df = df.filter(pl.col("created_datetime") > filter_datetime)
        
        return df



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
            self._vector_registry.load()
        return self._vector_registry

    def switch_collection(self, collection: str = "main") -> None:
        """Switch to a different collection, saving current state first."""
        if self._vector_registry is not None:
            self._vector_registry.save()
        
        self._vector_registry = VectorRegistry(self.path)
        self._vector_registry.load(collection)
        logger.info(f"Switched to collection '{collection}'")

    def insert(self, text: str, payload: Dict[str, Any]) -> None:
        """Embed text and store with payload."""
        vector = self.embedder.encode(text).tolist()
        payload["text"] = text
        
        self.vector_registry.vectors.append(vector)
        self.vector_registry.payloads.append(payload)
        self.vector_registry.created_datetime.append(dt.now())
        
        if self.vector_registry.origin_datetime is None:
            self.vector_registry.origin_datetime = dt.now()
        
        self.vector_registry.save()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search using cosine similarity."""
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
            result = {
                "score": round(float(scores[i]), 4),
                "text": self.vector_registry.payloads[i]["text"],
            }
            result.update({k: v for k, v in self.vector_registry.payloads[i].items() if k != "text"})
            results.append(result)
        
        return results

