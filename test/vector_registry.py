"""
Tests for VectorRegistry class using HDF5 storage.
"""

import os
import json
import pytest
import numpy as np
import h5py
from pathlib import Path

from axis_python.axis_db import VectorRegistry


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture that provides a temporary database path."""
    return str(tmp_path / "test_registry.db")


@pytest.fixture
def sample_vectors():
    """Fixture providing sample vector data."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3]
    ]


@pytest.fixture
def sample_payloads():
    """Fixture providing sample payload data."""
    return [
        {"text": "Hello world", "id": 1, "source": "test"},
        {"text": "Test payload", "id": 2, "source": "test"},
        {"text": "Another test", "id": 3, "source": "test"}
    ]


class TestVectorRegistryInitialization:
    """Tests for VectorRegistry initialization."""

    def test_init_creates_registry(self, temp_db_path):
        """Test that VectorRegistry initializes correctly."""
        registry = VectorRegistry(temp_db_path)
        
        assert registry.path == temp_db_path
        assert registry.collection_name == "main"
        assert registry.vectors == []
        assert registry.created_datetime == []
        assert registry.origin_datetime is None
        assert registry._payload_cache == {}

    def test_init_with_custom_collection(self, temp_db_path):
        """Test initialization with custom collection name."""
        registry = VectorRegistry(temp_db_path)
        registry.lazy_load("custom_collection")
        
        assert registry.collection_name == "custom_collection"


class TestVectorRegistrySaveAndLoad:
    """Tests for saving and loading vectors and payloads."""

    def test_save_creates_hdf5_file(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that save() creates a valid HDF5 file."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        
        registry.save()
        
        assert os.path.exists(temp_db_path)
        
        # Verify file structure
        with h5py.File(temp_db_path, 'r') as f:
            assert 'main' in f
            assert 'vectors' in f['main']
            assert 'payloads' in f['main']

    def test_save_correct_vector_count(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that saved vectors match original count."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        
        registry.save()
        
        with h5py.File(temp_db_path, 'r') as f:
            saved_vectors = f['main/vectors']
            assert saved_vectors.shape[0] == len(sample_vectors)
            assert saved_vectors.shape[1] == 4  # 4-dim vectors

    def test_save_correct_payload_count(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that saved payloads match original count."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        
        registry.save()
        
        with h5py.File(temp_db_path, 'r') as f:
            saved_payloads = f['main/payloads']
            assert saved_payloads.shape[0] == len(sample_payloads)

    def test_lazy_load_vectors(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that lazy_load correctly loads vectors."""
        # Create a file first
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        registry.save()
        
        # Load in a new instance
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        assert len(registry2.vectors) == len(sample_vectors)
        assert np.allclose(registry2.vectors[0], sample_vectors[0])

    def test_lazy_load_nonexistent_file(self, temp_db_path):
        """Test lazy_load with nonexistent file."""
        registry = VectorRegistry(temp_db_path)
        registry.lazy_load()
        
        assert registry.vectors == []

    def test_lazy_load_nonexistent_collection(self, temp_db_path, sample_vectors, sample_payloads):
        """Test lazy_load with nonexistent collection."""
        # Create a file with one collection
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        registry.save()
        
        # Try to load different collection
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load("nonexistent_collection")
        
        assert registry2.vectors == []


class TestVectorRegistryPayloads:
    """Tests for payload retrieval."""

    def test_get_payload_at_index(self, temp_db_path, sample_vectors, sample_payloads):
        """Test retrieving a specific payload by index."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        payload = registry2.get_payload_at_index(0)
        assert payload["text"] == "Hello world"
        assert payload["id"] == 1

    def test_get_payload_out_of_range(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that out-of-range index raises IndexError."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        with pytest.raises(IndexError):
            registry2.get_payload_at_index(999)

    def test_get_payload_negative_index(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that negative index raises IndexError."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        with pytest.raises(IndexError):
            registry2.get_payload_at_index(-1)

    def test_payload_cache(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that payloads are cached after first retrieval."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = sample_vectors
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        # First retrieval
        payload1 = registry2.get_payload_at_index(0)
        assert 0 in registry2._payload_cache
        
        # Second retrieval should come from cache
        payload2 = registry2.get_payload_at_index(0)
        assert payload1 == payload2


class TestVectorRegistryMultipleCollections:
    """Tests for handling multiple collections."""

    def test_multiple_collections_in_one_file(self, temp_db_path):
        """Test storing and retrieving multiple collections."""
        vectors1 = [[0.1, 0.2], [0.3, 0.4]]
        payloads1 = [{"id": 1}, {"id": 2}]
        
        vectors2 = [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        payloads2 = [{"id": 10}, {"id": 20}, {"id": 30}]
        
        # Save collection 1
        registry1 = VectorRegistry(temp_db_path)
        registry1.collection_name = "collection1"
        registry1.vectors = vectors1
        registry1._insertion_vector = payloads1
        registry1.save()
        
        # Save collection 2
        registry2 = VectorRegistry(temp_db_path)
        registry2.collection_name = "collection2"
        registry2.vectors = vectors2
        registry2._insertion_vector = payloads2
        registry2.save()
        
        # Load collection 1
        registry_load1 = VectorRegistry(temp_db_path)
        registry_load1.lazy_load("collection1")
        assert len(registry_load1.vectors) == 2
        
        # Load collection 2
        registry_load2 = VectorRegistry(temp_db_path)
        registry_load2.lazy_load("collection2")
        assert len(registry_load2.vectors) == 3

    def test_switch_collection(self, temp_db_path):
        """Test switching between collections."""
        registry = VectorRegistry(temp_db_path)
        
        # Save to collection1
        registry.collection_name = "collection1"
        registry.vectors = [[0.1, 0.2]]
        registry._insertion_vector = [{"id": 1}]
        registry.save()
        
        # Switch to collection2
        registry.collection_name = "collection2"
        registry.vectors = [[0.3, 0.4], [0.5, 0.6]]
        registry._insertion_vector = [{"id": 2}, {"id": 3}]
        registry.save()
        
        assert registry.collection_name == "collection2"


class TestVectorRegistryProperties:
    """Tests for VectorRegistry properties."""

    def test_payload_count(self, sample_vectors, sample_payloads):
        """Test payload_count property."""
        registry = VectorRegistry("dummy.db")
        registry.vectors = sample_vectors
        
        assert registry.payload_count == len(sample_vectors)

    def test_payload_count_empty(self):
        """Test payload_count with empty vectors."""
        registry = VectorRegistry("dummy.db")
        
        assert registry.payload_count == 0


class TestVectorRegistryDataTypes:
    """Tests for proper data type handling."""

    def test_float32_vectors(self, temp_db_path):
        """Test that vectors are stored as float32."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"id": 1}, {"id": 2}]
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = vectors
        registry._insertion_vector = payloads
        registry.save()
        
        with h5py.File(temp_db_path, 'r') as f:
            assert f['main/vectors'].dtype == np.float32

    def test_payload_encoding(self, temp_db_path):
        """Test that payloads are properly JSON encoded."""
        payloads = [{"text": "test", "nested": {"key": "value"}}]
        vectors = [[0.1, 0.2]]
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = vectors
        registry._insertion_vector = payloads
        registry.save()
        
        # Load and verify
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        retrieved = registry2.get_payload_at_index(0)
        assert retrieved["nested"]["key"] == "value"


class TestVectorRegistryLargeDatasets:
    """Tests for handling larger datasets."""

    def test_large_vector_count(self, temp_db_path):
        """Test saving and loading a large number of vectors."""
        n_vectors = 1000
        vector_dim = 384
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.random.rand(n_vectors, vector_dim).tolist()
        registry._insertion_vector = [{"id": i} for i in range(n_vectors)]
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        assert len(registry2.vectors) == n_vectors

    def test_large_payload_content(self, temp_db_path):
        """Test saving and retrieving large payload content."""
        large_text = "x" * 10000  # 10KB of text
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = [[0.1, 0.2]]
        registry._insertion_vector = [{"id": 1, "large_content": large_text}]
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        payload = registry2.get_payload_at_index(0)
        assert len(payload["large_content"]) == 10000
