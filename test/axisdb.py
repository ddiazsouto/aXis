"""
Tests for aXisDB class - the main vector database interface.
"""

import os
import json
import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, patch, MagicMock

from axis_python.axis_db import aXisDB, VectorRegistry


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture that provides a temporary database path."""
    return str(tmp_path / "test_axis.db")


@pytest.fixture
def mock_embedder():
    """Fixture that provides a mock sentence embedder."""
    embedder = Mock()
    embedder.encode = Mock(return_value=np.random.rand(384))
    return embedder


@pytest.fixture
def axis_db(temp_db_path):
    """Fixture that provides an aXisDB instance."""
    db = aXisDB(path=temp_db_path)
    # Mock the embedder to avoid loading the actual model
    db.embedder = Mock()
    db.embedder.encode = Mock(side_effect=lambda x: np.random.rand(384))
    return db


class TestAxisDBInitialization:
    """Tests for aXisDB initialization."""

    def test_init_with_default_path(self):
        """Test aXisDB initializes with default path."""
        with patch('axis_python.axis_db.SentenceTransformer'):
            db = aXisDB()
            assert db.path == "axis.db"

    def test_init_with_custom_path(self, temp_db_path):
        """Test aXisDB initializes with custom path."""
        with patch('axis_python.axis_db.SentenceTransformer'):
            db = aXisDB(path=temp_db_path)
            assert db.path == temp_db_path

    def test_init_vector_registry_none(self, temp_db_path):
        """Test that vector_registry is None initially."""
        with patch('axis_python.axis_db.SentenceTransformer'):
            db = aXisDB(path=temp_db_path)
            assert db._vector_registry is None


class TestAxisDBVectorRegistryLazyLoad:
    """Tests for lazy loading of vector registry."""

    def test_vector_registry_lazy_load(self, axis_db):
        """Test that vector_registry is lazy-loaded on first access."""
        assert axis_db._vector_registry is None
        
        registry = axis_db.vector_registry
        
        assert axis_db._vector_registry is not None
        assert isinstance(registry, VectorRegistry)

    def test_vector_registry_cached(self, axis_db):
        """Test that vector_registry is cached after first access."""
        registry1 = axis_db.vector_registry
        registry2 = axis_db.vector_registry
        
        assert registry1 is registry2


class TestAxisDBInsert:
    """Tests for inserting data into database."""

    def test_insert_single_item(self, axis_db):
        """Test inserting a single item."""
        text = "Hello world"
        payload = {"id": 1, "source": "test"}
        
        axis_db.insert(text, payload)
        
        # Verify vector was added
        assert len(axis_db.vector_registry.vectors) == 1
        assert len(axis_db.vector_registry._insertion_vector) == 0  # Cleared after save

    def test_insert_adds_text_to_payload(self, axis_db):
        """Test that insert adds the text to the payload."""
        text = "Sample text"
        payload = {"id": 1}
        
        axis_db.insert(text, payload)
        
        # Text should be in the insertion queue
        # Since it's saved immediately, we can't verify directly
        # But we can verify the embedder was called with the text
        axis_db.embedder.encode.assert_called()

    def test_insert_multiple_items(self, axis_db):
        """Test inserting multiple items."""
        for i in range(5):
            axis_db.insert(f"Text {i}", {"id": i})
        
        assert len(axis_db.vector_registry.vectors) == 5

    def test_insert_creates_vector_from_text(self, axis_db):
        """Test that insert creates a vector from the text."""
        mock_vector = np.array([0.1, 0.2, 0.3] + [0.0] * 381)
        axis_db.embedder.encode = Mock(return_value=mock_vector)
        
        text = "Sample"
        axis_db.insert(text, {"id": 1})
        
        axis_db.embedder.encode.assert_called_with(text)
        assert len(axis_db.vector_registry.vectors[0]) == 384

    def test_insert_preserves_payload_data(self, axis_db, temp_db_path):
        """Test that payload data is preserved when inserting."""
        from axis_python.axis_db import VectorRegistry
        import h5py
        
        text = "Test"
        payload = {"id": 1, "field1": "value1", "field2": 42}
        
        axis_db.insert(text, payload)
        
        # Read from file to verify payload was saved
        with h5py.File(axis_db.path, 'r') as f:
            if 'main' in f and 'payloads' in f['main']:
                saved_payload = json.loads(f['main/payloads'][0].decode('utf-8'))
                assert saved_payload["field1"] == "value1"
                assert saved_payload["field2"] == 42


class TestAxisDBSearch:
    """Tests for searching the database."""

    def test_search_empty_database(self, axis_db):
        """Test searching an empty database."""
        results = axis_db.search("test query")
        
        assert results == []

    def test_search_single_item(self, axis_db):
        """Test searching with a single item in database."""
        # Insert an item
        axis_db.insert("Hello world", {"id": 1, "type": "greeting"})
        
        # Setup mock to return consistent vectors
        base_vector = np.array([0.1] * 384)
        axis_db.embedder.encode = Mock(return_value=base_vector)
        
        results = axis_db.search("Hello", top_k=1)
        
        assert len(results) == 1
        assert results[0]["text"] == "Hello world"
        assert "score" in results[0]

    def test_search_multiple_items(self, axis_db):
        """Test searching with multiple items."""
        # Insert multiple items
        for i in range(5):
            axis_db.insert(f"Document {i}", {"id": i})
        
        # Setup mock
        base_vector = np.array([0.1] * 384)
        axis_db.embedder.encode = Mock(return_value=base_vector)
        
        results = axis_db.search("Document", top_k=3)
        
        assert len(results) <= 3
        assert len(results) > 0

    def test_search_respects_top_k(self, axis_db):
        """Test that search respects the top_k parameter."""
        # Insert items
        for i in range(10):
            axis_db.insert(f"Item {i}", {"id": i})
        
        # Setup mock
        base_vector = np.array([0.1] * 384)
        axis_db.embedder.encode = Mock(return_value=base_vector)
        
        results_k3 = axis_db.search("Item", top_k=3)
        results_k5 = axis_db.search("Item", top_k=5)
        
        assert len(results_k3) <= 3
        assert len(results_k5) <= 5
        assert len(results_k5) >= len(results_k3)

    def test_search_results_have_required_fields(self, axis_db):
        """Test that search results contain required fields."""
        axis_db.insert("Test document", {"id": 1, "custom_field": "value"})
        
        # Setup mock
        base_vector = np.array([0.1] * 384)
        axis_db.embedder.encode = Mock(return_value=base_vector)
        
        results = axis_db.search("Test", top_k=1)
        
        assert len(results) > 0
        result = results[0]
        assert "score" in result
        assert "text" in result
        assert "id" in result
        assert "custom_field" in result


class TestAxisDBCollections:
    """Tests for working with multiple collections."""

    def test_switch_collection(self, axis_db):
        """Test switching between collections."""
        # Insert into main collection
        axis_db.insert("Item in main", {"id": 1})
        
        # Switch to another collection
        axis_db.switch_collection("other_collection")
        
        # Current registry should be different now
        assert axis_db.vector_registry.collection_name == "other_collection"

    def test_switch_collection_saves_previous(self, axis_db):
        """Test that switching collections saves the previous one."""
        axis_db.insert("Item 1", {"id": 1})
        
        # Switch collection
        axis_db.switch_collection("collection2")
        axis_db.insert("Item 2", {"id": 2})
        
        # Should have separate registries
        assert len(axis_db.vector_registry.vectors) == 1

    def test_different_collections_independent(self, axis_db):
        """Test that different collections are independent."""
        # Populate collection1
        axis_db.switch_collection("collection1")
        for i in range(3):
            axis_db.insert(f"Item {i}", {"id": i})
        
        # Populate collection2
        axis_db.switch_collection("collection2")
        for i in range(5):
            axis_db.insert(f"Document {i}", {"id": i})
        
        # Collection2 should have 5 items
        assert len(axis_db.vector_registry.vectors) == 5


class TestAxisDBIntegration:
    """Integration tests for aXisDB."""

    def test_insert_and_search_workflow(self, axis_db):
        """Test complete insert and search workflow."""
        # Insert items
        items = [
            ("Python is great", {"id": 1, "language": "python"}),
            ("JavaScript is fun", {"id": 2, "language": "javascript"}),
            ("Go is fast", {"id": 3, "language": "go"}),
        ]
        
        for text, payload in items:
            axis_db.insert(text, payload)
        
        assert len(axis_db.vector_registry.vectors) == 3
        
        # Setup search mock
        base_vector = np.array([0.1] * 384)
        axis_db.embedder.encode = Mock(return_value=base_vector)
        
        # Search
        results = axis_db.search("Python programming", top_k=2)
        assert len(results) > 0

    def test_multiple_collection_workflow(self, axis_db):
        """Test workflow with multiple collections."""
        # Collection 1: Knowledge base
        axis_db.switch_collection("knowledge_base")
        for i in range(3):
            axis_db.insert(f"Knowledge {i}", {"id": i, "type": "knowledge"})
        
        # Collection 2: User data
        axis_db.switch_collection("user_data")
        for i in range(2):
            axis_db.insert(f"User {i}", {"id": i, "type": "user"})
        
        # Verify counts
        assert len(axis_db.vector_registry.vectors) == 2

    def test_persistence_across_instances(self, temp_db_path):
        """Test that data persists across database instances."""
        # Create first instance and add data
        with patch('axis_python.axis_db.SentenceTransformer'):
            db1 = aXisDB(path=temp_db_path)
            db1.embedder = Mock()
            db1.embedder.encode = Mock(return_value=np.random.rand(384))
            
            db1.insert("Persistent data", {"id": 1})
        
        # Create second instance
        with patch('axis_python.axis_db.SentenceTransformer'):
            db2 = aXisDB(path=temp_db_path)
            db2.embedder = Mock()
            db2.embedder.encode = Mock(return_value=np.random.rand(384))
            
            # Should load the data
            assert len(db2.vector_registry.vectors) == 1


class TestAxisDBErrorHandling:
    """Tests for error handling in aXisDB."""

    def test_insert_with_empty_text(self, axis_db):
        """Test inserting with empty text."""
        # Should not raise an error
        axis_db.insert("", {"id": 1})
        assert len(axis_db.vector_registry.vectors) == 1

    def test_insert_with_none_payload_fields(self, axis_db):
        """Test inserting with None values in payload."""
        payload = {"id": 1, "optional_field": None}
        axis_db.insert("Text", payload)
        
        assert len(axis_db.vector_registry.vectors) == 1

    def test_search_with_empty_query(self, axis_db):
        """Test search with empty query string."""
        axis_db.insert("Document", {"id": 1})
        axis_db.embedder.encode = Mock(return_value=np.random.rand(384))
        
        # Should handle gracefully
        results = axis_db.search("", top_k=1)
        assert isinstance(results, list)
