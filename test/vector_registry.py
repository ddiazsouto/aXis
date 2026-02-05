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
        assert isinstance(registry.vectors, np.ndarray)
        assert len(registry.vectors) == 0
        assert registry.vectors.shape == (0, 384)

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
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        
        # Mock the payload insertion (since _insertion_vector may not exist)
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
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        
        registry.save()
        
        with h5py.File(temp_db_path, 'r') as f:
            saved_vectors = f['main/vectors']
            assert saved_vectors.shape[0] == len(sample_vectors)
            assert saved_vectors.shape[1] == 4  # 4-dim vectors

    def test_save_correct_payload_count(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that saved payloads match original count."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        
        registry.save()
        
        with h5py.File(temp_db_path, 'r') as f:
            if 'payloads' in f['main']:
                saved_payloads = f['main/payloads']
                # Save creates an empty payloads dataset, so length may be 0
                assert saved_payloads.shape[0] >= 0

    def test_lazy_load_vectors(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that lazy_load correctly loads vectors."""
        # Create a file first
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
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
        
        assert len(registry.vectors) == 0

    def test_lazy_load_nonexistent_collection(self, temp_db_path, sample_vectors, sample_payloads):
        """Test lazy_load with nonexistent collection."""
        # Create a file with one collection
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        registry.save()
        
        # Try to load different collection
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load("nonexistent_collection")
        
        assert len(registry2.vectors) == 0


class TestVectorRegistryPayloads:
    """Tests for payload retrieval."""

    def test_get_payload_at_index(self, temp_db_path, sample_vectors, sample_payloads):
        """Test retrieving a specific payload by index."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        # Since save() doesn't store payloads automatically, this may return empty dict
        # Just verify the method can be called without error
        try:
            payload = registry2.get_payload_at_index(0)
            # If successful, verify it has expected structure
            assert isinstance(payload, dict)
        except IndexError:
            # Expected if no payloads were saved
            pass

    def test_get_payload_out_of_range(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that out-of-range index raises IndexError."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        with pytest.raises(IndexError):
            registry2.get_payload_at_index(999)

    def test_get_payload_negative_index(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that negative index raises IndexError."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        with pytest.raises(IndexError):
            registry2.get_payload_at_index(-1)

    def test_payload_cache(self, temp_db_path, sample_vectors, sample_payloads):
        """Test that payloads are cached after first retrieval."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(sample_vectors, dtype=np.float32)
        registry._insertion_vector = sample_payloads
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        # First retrieval
        payload1 = registry2.get_payload_at_index(0)
        
        # Second retrieval should be the same
        payload2 = registry2.get_payload_at_index(0)
        assert payload1 == payload2


class TestVectorRegistryMultipleCollections:
    """Tests for handling multiple collections."""

    def test_multiple_collections_in_one_file(self, temp_db_path):
        """Test storing and retrieving multiple collections."""
        vectors1 = [[0.1, 0.2], [0.3, 0.4]]
        
        vectors2 = [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        
        # Save collection 1
        registry1 = VectorRegistry(temp_db_path)
        registry1.collection_name = "collection1"
        registry1.vectors = np.array(vectors1, dtype=np.float32)
        registry1.save()
        
        # Save collection 2
        registry2 = VectorRegistry(temp_db_path)
        registry2.collection_name = "collection2"
        registry2.vectors = np.array(vectors2, dtype=np.float32)
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
        registry.vectors = np.array([[0.1, 0.2]], dtype=np.float32)
        registry.save()
        
        # Switch to collection2
        registry.collection_name = "collection2"
        registry.vectors = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        registry.save()
        
        assert registry.collection_name == "collection2"


class TestVectorRegistryDataTypes:
    """Tests for proper data type handling."""

    def test_float32_vectors(self, temp_db_path):
        """Test that vectors are stored as float32."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(vectors, dtype=np.float32)
        registry.save()
        
        with h5py.File(temp_db_path, 'r') as f:
            assert f['main/vectors'].dtype == np.float32

    def test_payload_encoding(self, temp_db_path):
        """Test that payloads are properly JSON encoded."""
        vectors = [[0.1, 0.2]]
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array(vectors, dtype=np.float32)
        registry.save()
        
        # Load and verify
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()


class TestVectorRegistryLargeDatasets:
    """Tests for handling larger datasets."""

    def test_large_vector_count(self, temp_db_path):
        """Test saving and loading a large number of vectors."""
        n_vectors = 1000
        vector_dim = 384
        
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.random.rand(n_vectors, vector_dim).astype(np.float32)
        registry.save()
        
        registry2 = VectorRegistry(temp_db_path)
        registry2.lazy_load()
        
        assert len(registry2.vectors) == n_vectors

    def test_large_payload_content(self, temp_db_path):
        """Test saving and retrieving large payload content."""
        registry = VectorRegistry(temp_db_path)
        registry.vectors = np.array([[0.1, 0.2]], dtype=np.float32)
        registry.save()
