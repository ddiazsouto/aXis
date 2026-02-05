import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import os
from datetime import datetime as dt

import h5py


logger = logging.getLogger(__name__)


class VectorRegistry:

    def __init__(self, path: str):
        self.path = path
        self.collection_name: str = "main"
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)  # Start with empty array
        self.created_datetime: List[dt] = []
        self.origin_datetime: Optional[dt] = None
        self._payload_cache: Dict[int, Dict[str, Any]] = {}

    def lazy_load(self, collection: str = "main") -> None:
        """
        Description:
            Load vectors from HDF5 using optimized settings for fastest access.
        
        Args:
            collection: str
                Name of the collection to load (default: "main")
        """
        self.collection_name = collection
        if not os.path.exists(self.path):
            logger.info(f"Database file {self.path} does not exist yet, starting with empty data")
            return

        try:
            with h5py.File(self.path, 'r') as f:
                group = f.get(collection)
                if group is None:
                    logger.info(f"Collection '{collection}' not found or empty, starting with empty data")
                    return
                
                if 'vectors' in group:
                    vectors_ds = group['vectors']
                    self.vectors = vectors_ds[:].astype(np.float32)
                    logger.info(f"Loaded {len(self.vectors)} vectors from {self.path} (contiguous, no decompression)")
                else:
                    logger.info(f"No vectors found in collection '{collection}'")
            
        except Exception as e:
            logger.error(f"Failed to load {self.path}: {e}")
            raise

    def get_payload_at_index(self, payload_index: int) -> Dict[str, Any]:
        """
        Description:
            Retrieve the payload after finding its vector index.
        
        Args:
            payload_index: int
                The index of the payload in the vectors array (same index as the vector).

        Returns:
            payload: Dict[str, Any]
                The payload associated with the vector at the given index.
        """
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
        """
        Description:
            Save vectors and payloads to HDF5 with optimized settings for vector speed.
        """
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
                
                all_payloads.extend(insertions)
                
                if 'vectors' in group:
                    del group['vectors']
                if 'payloads' in group:
                    del group['payloads']
                
                if isinstance(self.vectors, list):
                    all_vectors = np.array(self.vectors, dtype=np.float32)
                else:
                    all_vectors = self.vectors.astype(np.float32) if self.vectors.dtype != np.float32 else self.vectors
                group.create_dataset(
                    'vectors',
                    data=all_vectors,
                    chunks=None,
                    compression=None,
                    shuffle=False
                )
                payload_strings = [json.dumps(p).encode('utf-8') for p in all_payloads]
                dt_string = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(
                    'payloads',
                    data=payload_strings,
                    dtype=dt_string,
                    compression="gzip",
                    compression_opts=4
                )
                
                if self.created_datetime:
                    group.attrs['created_datetime'] = [d.isoformat() for d in self.created_datetime]
                if self.origin_datetime:
                    group.attrs['origin_datetime'] = self.origin_datetime.isoformat()
            
            logger.debug(f"Saved collection '{self.collection_name}' to {self.path} (optimized for vector speed)")
        
        except Exception as e:
            logger.error(f"Failed to save {self.path}: {e}")
            raise
