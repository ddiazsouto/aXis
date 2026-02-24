import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import os
import polars as pl
from datetime import datetime


from axis_python.vector_registry import VectorRegistry
from axis_python.transformers import BertModel


class aXisDB:
    """Vector database with semantic search capabilities."""

    def __init__(self, path: str = "main"):
        self.path = path
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'all-MiniLM-L6-v2')
        self.embedder = BertModel(model_path)
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
        payload = str(payload)
        
        df = pl.DataFrame({
            "payload": [payload],
            "text": [text],
            "vector": [vector],
            "timestamp": [datetime.now()]
        })
        
        self.vector_registry.insertion_matrix.append(
            df.with_row_index("index", offset=self.vector_registry.vectors_count)
        )
        self.logger.info(f"Inserted new record with text: '{text}' and payload: {payload} and index: {self.vector_registry.vectors_count}")

    
    def insert_dataframe(
        self,
        dataframe: pl.DataFrame,
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
                )
            ).alias("vector"),
            pl.lit(datetime.now()).alias("timestamp")
        )
        renamed_df = payload_df.rename(
            {payload_col: "payload", vectorise_col: "text"}
        ).select(["payload", "vector", "text", "timestamp"])
        print("staging a dataframe with the following number of records", renamed_df.height)
        self.vector_registry.insertion_matrix.append(
            renamed_df.with_row_index("index", offset=self.vector_registry.vectors_count)
        )

    def search(
        self,
        query: str,
        show_embedded_text: bool = False,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Description:
            Semantic search using cosine similarity, retrieving payloads on-demand.
        """
        query_norm = self.embedder.encode(
            query,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        tvector = self.vector_registry.vectors.collect() if hasattr(self.vector_registry.vectors, 'collect') else self.vector_registry.vectors

        vectors_matrix = np.vstack(
            tvector["vector"].to_list()
        )
        similarities = vectors_matrix @ query_norm
        
        top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_k_sorted_idx = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        original_indices = tvector["index"].to_numpy()[top_k_sorted_idx]
        top_matches_dataframe = [
            (self.vector_registry.get_payload_at_index(i), top_k_indices)
            for i in original_indices.tolist()
        ]
        
        return [
            (dataframe[0]["payload"][0],
             dataframe[0]["text"][0] if show_embedded_text else None,
             dataframe[0]["index"][0])
            for dataframe in top_matches_dataframe
        ]
