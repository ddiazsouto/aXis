import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
import polars as pl
from datetime import datetime


from axis_python.functions.cosine_similarity import cosine_similarity
from axis_python.vector_registry import VectorRegistry



class aXisDB:
    """Vector database with semantic search capabilities."""

    def __init__(self, path: str = "main"):
        if not path.endswith('.parquet'):
            path = f"{path}.parquet"
        
        self.path = path
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(model_path, local_files_only=True)
        self._vector_registry: Optional[VectorRegistry] = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"aXisDB initialized with database: {self.path}")

    @property
    def vector_registry(self) -> VectorRegistry:
        """Lazy-load vector registry on first access."""
        if self._vector_registry is None:
            self._vector_registry = VectorRegistry(self.path)
            self._vector_registry.lazy_load()
        return self._vector_registry

    def insert(self, text: str, payload: Dict[str, Any]) -> None:
        """
        Description:
            Embed text and store with payload.
        """
        vector = self.embedder.encode(
            text,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        payload["text"] = text
        
        if len(self.vector_registry.vectors) == 0:
            self.vector_registry.vectors = vector.reshape(1, -1)
        else:
            self.vector_registry.vectors = np.vstack([self.vector_registry.vectors, vector])
        
        self.vector_registry._insertion_matrix.append(payload)
        
        self.vector_registry.save()
    
    def insert_dataframe(
        self, dataframe: pl.DataFrame,
        vectorise_col: str,
        payload_col: str
    ) -> None:
        """
        Description:
            Insert a Polars DataFrame with 'text', 'payload' and indexcolumns.
        """
        if vectorise_col not in dataframe.columns or payload_col not in dataframe.columns:
            raise ValueError(f"DataFrame must contain '{vectorise_col}' and '{payload_col}' columns")
        payload_df = dataframe.with_columns(
            pl.col(vectorise_col)
            .map_batches(
                lambda texts: self.embedder.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype(np.float32)
            )
            .alias("vector"),
            pl.lit(datetime.now()).alias("timestamp")
        )
        renamed_df = payload_df.rename(
            {payload_col: "payload", vectorise_col: "text"}
        ).select(["payload", "vector", "timestamp"])
        print("staging a dataframe with the following number of records", renamed_df.height)
        self.vector_registry._insertion_matrix.append(renamed_df)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Description:
            Semantic search using cosine similarity, retrieving payloads on-demand.
        """
        
        query_norm = query.astype(np.float32)
        
        similarities = self.embeddings @ query_norm   # dot product
        
        top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_k_sorted_idx = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        top_similarities = similarities[top_k_sorted_idx]
        top_original_indices = self.indices[top_k_sorted_idx]

        payload_df = (
            self.vectors
            .filter(pl.col("index").is_in(top_original_indices.tolist()))
            .collect()
        )
