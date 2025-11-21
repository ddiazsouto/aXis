import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from dataclasses import dataclass
import datetime

import polars as pl

from axis_python.functions.cosine_similarity import cosine_similarity



class VectorRegistry:

    vectors: List[List[float]]
    input: List[str]
    created_datetime: List[datetime.datetime]
    payloads: List[Dict[str, Any]]
    origin_datetime: datetime.datetime
    last_updated_datetime: datetime.datetime
    collection_name: str

    def __init__(self, path: str):
        self.path = path
        self.collection_name = None

    def load(self, collection: str = "main"):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                collection = data.get(collection, None)
                if collection is None:
                    raise ValueError(f"Collection {collection} not found in {self.path}")    
                self.vectors = collection.get("vectors", [])
                self.payloads = collection.get("payloads", [])
                self.origin_datetime = collection.get("origin_datetime", None)
                self.collection_name = collection

                print(f"Loaded {len(self.vector_registry.vectors)} vectors from {self.path}")
            except Exception as e:
                print(f"Failed to load {self.path}: {e}")

    def save(self):
        collection_data = {
            "input": self.input,
            "created_datetime": self.created_datetime,
            "vectors": self.vector_registry.vectors,
            "payloads": self.vector_registry.payloads
        }
        with open(self.path, "r") as f:
            data = json.load(f)

        data[self.collection_name].update(collection_data)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    
    def show_collection_data(self, start_datetime: datetime.datetime = None):

        start_datetime = start_datetime
        if start_datetime is None:
            start_datetime = self.origin_datetime

        df = pl.DataFrame({
            "input": self.input,
            "payload": self.payloads,
            "created_datetime": self.created_datetime,
        }).where(
            pl.col("created_datetime") > datetime.datetime.now()
        )



class aXisDB:

    def __init__(self, path: str = "axis.db"):
        self.path = path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._vector_registry = None

    @property
    def vector_registry(self) -> VectorRegistry:
        if self._vector_registry is None:
            self._vector_registry = VectorRegistry(self.path)
            self._vector_registry.load()

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

