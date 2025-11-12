import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from dataclasses import dataclass
import datetime

from axis_python.functions.cosine_similarity import cosine_similarity


@dataclass
class vector_registry:
    vectors: List[List[float]]
    created_datetime: List[datetime.datetime]
    payloads: List[Dict[str, Any]]
    as_timestamp: datetime.datetime

class aXisDB:
    def __init__(self, path: str = "axis.db"):
        self.path = path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._vector_registry = None
        self.load()    

    @property
    def vector_registry(self) -> vector_registry:
        if self._vector_registry is None:
            self._vector_registry = vector_registry(
                vectors=[],
                created_datetime=[],
                payloads=[],
                as_timestamp=datetime.datetime.now()
            )
        return self._vector_registry

    def insert(self, text: str, payload: Dict[str, Any]):
        """Embed text and store with payload"""
        vector = self.embedder.encode(text).tolist()
        payload = payload
        payload["text"] = text
        self.vector_registry.vectors.append(vector)
        self.vector_registry.payloads.append(payload)
        self.save()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic search using cosine similarity"""
        if not self.vector_registry.vectors:
            return []
        q_vec = self.embedder.encode(query)
        scores = [
            cosine_similarity(q_vec, np.array(v))
            for v in self.vector_registry.vectors
        ]
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [
            {
                "score": round(float(scores[i]), 4),
                "text": self.vector_registry.payloads[i]["text"],
                **{k: v for k, v 
                   in self.vector_registry.payloads[i].items()
                   if k != "text"}
            }
            for i in top_idx
        ]

    def save(self):
        data = {
            "vectors": self.vector_registry.vectors,
            "payloads": self.vector_registry.payloads
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.vector_registry.vectors = data.get("vectors", [])
                self.vector_registry.payloads = data.get("payloads", [])
                print(f"Loaded {len(self.vector_registry.vectors)} vectors from {self.path}")
            except Exception as e:
                print(f"Failed to load {self.path}: {e}")