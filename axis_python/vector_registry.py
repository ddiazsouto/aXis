import json
import logging
import numpy as np
from typing import List, Dict, Any
import os
import polars as pl


logger = logging.getLogger(__name__)


class VectorRegistry:
    """
    Simple vector registry that stores vectors and payloads in Parquet format.
    Uses an index column for fast payload lookup.
    """

    def __init__(self, path: str, embedding_dim: int = 384):
        self.path = path
        self.embedding_dim = embedding_dim
        self.vectors: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self.payloads: List[Dict[str, Any]] = []

        self._pending_inserts: List[pl.DataFrame] = []

    @property
    def _insertion_matrix(self) -> List[pl.DataFrame]:
        """List of DataFrames waiting to be appended on save."""
        return self._pending_inserts

    @_insertion_matrix.setter
    def _insertion_matrix(self, df: pl.DataFrame) -> None:
        """
        Add a single Polars DataFrame to the pending inserts.
        Enforces the required schema.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Only polars DataFrame objects can be added to _insertion_matrix")

        expected_columns = {"payload", "index", "timestamp"}
        if set(df.columns) != expected_columns:
            raise ValueError(
                f"DataFrame must have exactly these columns: {expected_columns}. "
                f"Got: {set(df.columns)}"
            )
        self._pending_inserts.append(df)

    def lazy_load(self) -> None:
        """Load vectors and payloads from Parquet file."""
        if not os.path.exists(self.path):
            logger.info(f"Database file {self.path} does not exist, starting with empty data")
            return
        
        try:                      
            self.vectors = pl.scan_delta(self.path).select("vector",)
            self.payloads = pl.scan_delta(self.path).select("payload")

            self.vectors_count = self.payloads.select(pl.len()).collect().item()
            logger.info(f"Loaded {self.vectors_count} vectors from {self.path}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.path}: {e}")
            raise
    
    def registry_dot_product(self, normalised_vector: np.ndarray) -> np.ndarray:
        """
        Description:
            Compute dot product between input vector and all vectors in the registry.
            The vector input should be normalised    
        """
        return self.vectors @ normalised_vector

    def get_payload_at_index(self, index: int) -> Dict[str, Any]:
        """
        Retrieve the payload at a given index.
        
        Args:
            index: int - The index of the payload
            
        Returns:
            Dict[str, Any] - The payload at the given index
        """
        if index < 0 or index >= len(self.payloads):
            raise IndexError(f"Index {index} out of range [0, {len(self.payloads)})")
        
        return self.payloads[index]

    def get_vector_at_index(self, index: int) -> np.ndarray:
        """
        Retrieve the vector at a given index.
        
        Args:
            index: int - The index of the vector
            
        Returns:
            np.ndarray - The vector at the given index
        """
        if index < 0 or index >= len(self.vectors):
            raise IndexError(f"Index {index} out of range [0, {len(self.vectors)})")
        
        return self.vectors[index]

    def save(self) -> None:
        """Save vectors and payloads to Parquet with index column."""
        if len(self._insertion_matrix) == 0:
            logger.warning("No vectors to save")
            return
        
        try:            
 
            df = pl.concat(
                self._insertion_matrix
            )
            df.write_delta(self.path, mode="append")
            
            logger.info(f"Saved {len(self._insertion_matrix)} vectors to {self.path}")
            
        except Exception as e:
            logger.error(f"Failed to save {self.path}: {e}")
            raise

    def __len__(self) -> int:
        """Return the number of vectors in the registry."""
        return len(self.vectors)

    # def __getitem__(self, index: int) -> tuple:
    #     """Return (vector, payload) for the given index."""
    #     return self.get_vector_at_index(index), self.get_payload_at_index(index)
