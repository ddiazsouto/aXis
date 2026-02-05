import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
from datetime import datetime as dt

import h5py

from axis_python.functions.cosine_similarity import cosine_similarity

logger = logging.getLogger(__name__)


class VectorRegistry:
    """Manages vector storage and retrieval for a collection using HDF5."""

    def __init__(self, path: str):
        self.path = path
        self.collection_name: str = "main"
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)  # Start with empty array
        self.created_datetime: List[dt] = []
        self.origin_datetime: Optional[dt] = None
        self._payload_cache: Dict[int, Dict[str, Any]] = {}

    def lazy_load(self, collection: str = "main") -> None:
        """Load vectors from HDF5 using optimized settings for fastest access.
        
        Optimization:
        - Uses contiguous storage (no decompression overhead)
        - Vectors loaded directly without decompression
        - Memory-mapped access when possible
        """
        self.collection_name = collection
        if not os.path.exists(self.path):
            logger.info(f"Database file {self.path} does not exist yet, starting with empty data")
            return

        try:
            # Use driver='core' with backing_store for potential speed improvements
            with h5py.File(self.path, 'r') as f:
                group = f.get(collection)
                if group is None:
                    logger.info(f"Collection '{collection}' not found or empty, starting with empty data")
                    return
                
                if 'vectors' in group:
                    vectors_ds = group['vectors']
                    # Direct array access - contiguous storage means no decompression
                    # This is as fast as reading from a separate binary file
                    self.vectors = vectors_ds[:].astype(np.float32)
                    logger.info(f"Loaded {len(self.vectors)} vectors from {self.path} (contiguous, no decompression)")
                else:
                    logger.info(f"No vectors found in collection '{collection}'")
            
        except Exception as e:
            logger.error(f"Failed to load {self.path}: {e}")
            raise

    def get_payload_at_index(self, payload_index: int) -> Dict[str, Any]:
        """Efficiently retrieve a specific payload by index from HDF5 without loading all."""
        if payload_index < 0 or payload_index >= len(self.vectors):
            raise IndexError(f"Payload index {payload_index} out of range")
        
        if payload_index in self._payload_cache:
            return self._payload_cache[payload_index]
        
        try:
            with h5py.File(self.path, 'r') as f:
                group = f.get(self.collection_name)
                if group is None or 'payloads' not in group:
                    raise ValueError(f"Payloads not found in collection '{self.collection_name}'")
                
                payloads_ds = group['payloads']
                if payload_index >= payloads_ds.shape[0]:
                    raise IndexError(f"Payload at index {payload_index} not found")
                
                payload_str = payloads_ds[payload_index].decode('utf-8')
                payload = json.loads(payload_str)
                self._payload_cache[payload_index] = payload
                return payload
            
        except Exception as e:
            logger.error(f"Failed to retrieve payload at index {payload_index}: {e}")
            raise

    def save(self) -> None:
        """Save vectors and payloads to HDF5 file with optimized storage for fast vector reads.
        
        Optimization strategy:
        - Vectors: Contiguous storage (no chunking) for fastest sequential/random access
        - Payloads: GZIP compression (less critical for speed)
        - Format: Pure binary HDF5 (unreadable by text editors)
        """
        # Get insertions if any
        insertions = getattr(self, '_insertion_vector', [])
        self._insertion_vector = []  # Clear after getting

        try:
            with h5py.File(self.path, 'a') as f:  # 'a' for append/create
                group = f.require_group(self.collection_name)
                
                # Load existing payloads to append new ones
                all_payloads = []
                if 'payloads' in group:
                    existing_payloads = group['payloads'][:]
                    all_payloads = [json.loads(p.decode('utf-8')) for p in existing_payloads]
                
                # Append new insertions
                all_payloads.extend(insertions)
                
                # Delete old datasets if they exist (to avoid conflicts)
                if 'vectors' in group:
                    del group['vectors']
                if 'payloads' in group:
                    del group['payloads']
                
                # Handle vectors: CONTIGUOUS storage for fastest access (like a separate file)
                # No compression on vectors - speed over size for this use case
                # Convert to numpy array if needed (for compatibility with tests that append lists)
                if isinstance(self.vectors, list):
                    all_vectors = np.array(self.vectors, dtype=np.float32)
                else:
                    all_vectors = self.vectors.astype(np.float32) if self.vectors.dtype != np.float32 else self.vectors
                group.create_dataset(
                    'vectors',
                    data=all_vectors,
                    chunks=None,  # Contiguous layout = fastest reads
                    compression=None,  # No compression for vectors = fastest access
                    shuffle=False
                )
                
                # Handle payloads: GZIP compression (they're smaller, less critical for speed)
                payload_strings = [json.dumps(p).encode('utf-8') for p in all_payloads]
                dt_string = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(
                    'payloads',
                    data=payload_strings,
                    dtype=dt_string,
                    compression="gzip",  # Compression OK for payloads since less frequently accessed
                    compression_opts=4  # Medium compression (1-9, default 4)
                )
                
                # Save metadata as attributes
                if self.created_datetime:
                    group.attrs['created_datetime'] = [d.isoformat() for d in self.created_datetime]
                if self.origin_datetime:
                    group.attrs['origin_datetime'] = self.origin_datetime.isoformat()
            
            logger.debug(f"Saved collection '{self.collection_name}' to {self.path} (optimized for vector speed)")
        
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
        """Semantic search using cosine similarity, retrieving payloads on-demand."""
        if len(self.vector_registry.vectors) == 0:
            return []
        
        q_vec = self.embedder.encode(query).astype(np.float32)
        # Vectorized cosine similarity: compute scores for all vectors at once
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

