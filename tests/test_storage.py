"""Tests for RAG OS storage backends."""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

from rag_os.storage import (
    BaseStorage,
    BaseDocumentStorage,
    StorageConfig,
    StorageType,
    StoredItem,
    MemoryStorage,
    MemoryDocumentStorage,
    FileStorage,
    FileDocumentStorage,
)


# =============================================================================
# StoredItem Tests
# =============================================================================

class TestStoredItem:
    """Tests for StoredItem dataclass."""

    def test_create_stored_item(self):
        """Test creating a stored item."""
        item = StoredItem(key="test", value={"data": 123})

        assert item.key == "test"
        assert item.value == {"data": 123}
        assert item.created_at is not None
        assert item.updated_at is not None
        assert item.expires_at is None
        assert item.metadata == {}

    def test_item_not_expired(self):
        """Test item not expired without expiration."""
        item = StoredItem(key="test", value="value")

        assert item.is_expired is False

    def test_item_not_expired_with_future_time(self):
        """Test item not expired with future expiration."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        item = StoredItem(key="test", value="value", expires_at=future)

        assert item.is_expired is False

    def test_item_expired(self):
        """Test item expired with past expiration."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        item = StoredItem(key="test", value="value", expires_at=past)

        assert item.is_expired is True


# =============================================================================
# StorageConfig Tests
# =============================================================================

class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StorageConfig()

        assert config.storage_type == StorageType.MEMORY
        assert config.path == ""
        assert config.connection_string == ""
        assert config.namespace == "default"
        assert config.ttl_seconds is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = StorageConfig(
            storage_type=StorageType.FILE,
            path="/tmp/storage",
            namespace="test",
            ttl_seconds=3600,
        )

        assert config.storage_type == StorageType.FILE
        assert config.path == "/tmp/storage"
        assert config.namespace == "test"
        assert config.ttl_seconds == 3600


# =============================================================================
# MemoryStorage Tests
# =============================================================================

class TestMemoryStorage:
    """Tests for MemoryStorage backend."""

    def test_basic_get_set(self):
        """Test basic get and set operations."""
        storage = MemoryStorage()

        storage.set("key1", "value1")
        assert storage.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        storage = MemoryStorage()

        assert storage.get("nonexistent") is None

    def test_set_overwrites_existing(self):
        """Test that set overwrites existing values."""
        storage = MemoryStorage()

        storage.set("key1", "value1")
        storage.set("key1", "value2")

        assert storage.get("key1") == "value2"

    def test_delete_existing_key(self):
        """Test deleting an existing key."""
        storage = MemoryStorage()
        storage.set("key1", "value1")

        result = storage.delete("key1")

        assert result is True
        assert storage.get("key1") is None

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        storage = MemoryStorage()

        result = storage.delete("nonexistent")

        assert result is False

    def test_exists(self):
        """Test checking if key exists."""
        storage = MemoryStorage()
        storage.set("key1", "value1")

        assert storage.exists("key1") is True
        assert storage.exists("nonexistent") is False

    def test_keys_all(self):
        """Test listing all keys."""
        storage = MemoryStorage()
        storage.set("key1", "v1")
        storage.set("key2", "v2")
        storage.set("key3", "v3")

        keys = storage.keys()

        assert len(keys) == 3
        assert set(keys) == {"key1", "key2", "key3"}

    def test_keys_with_pattern(self):
        """Test listing keys with pattern."""
        storage = MemoryStorage()
        storage.set("user:1", "v1")
        storage.set("user:2", "v2")
        storage.set("order:1", "v3")

        user_keys = storage.keys("user:*")

        assert len(user_keys) == 2
        assert set(user_keys) == {"user:1", "user:2"}

    def test_clear(self):
        """Test clearing all items."""
        storage = MemoryStorage()
        storage.set("key1", "v1")
        storage.set("key2", "v2")

        count = storage.clear()

        assert count == 2
        assert storage.get("key1") is None
        assert storage.get("key2") is None

    def test_namespace_isolation(self):
        """Test namespace isolation."""
        config1 = StorageConfig(namespace="ns1")
        config2 = StorageConfig(namespace="ns2")

        storage1 = MemoryStorage(config1)
        storage2 = MemoryStorage(config2)

        storage1.set("key", "value1")
        storage2.set("key", "value2")

        assert storage1.get("key") == "value1"
        assert storage2.get("key") == "value2"

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        storage = MemoryStorage()
        storage.set("key1", "value1", ttl_seconds=1)

        # Value should exist initially
        assert storage.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Value should be expired
        assert storage.get("key1") is None
        assert storage.exists("key1") is False

    def test_get_many(self):
        """Test getting multiple keys."""
        storage = MemoryStorage()
        storage.set("key1", "v1")
        storage.set("key2", "v2")
        storage.set("key3", "v3")

        result = storage.get_many(["key1", "key3", "nonexistent"])

        assert result == {"key1": "v1", "key3": "v3"}

    def test_set_many(self):
        """Test setting multiple keys."""
        storage = MemoryStorage()

        storage.set_many({"key1": "v1", "key2": "v2", "key3": "v3"})

        assert storage.get("key1") == "v1"
        assert storage.get("key2") == "v2"
        assert storage.get("key3") == "v3"

    def test_delete_many(self):
        """Test deleting multiple keys."""
        storage = MemoryStorage()
        storage.set("key1", "v1")
        storage.set("key2", "v2")
        storage.set("key3", "v3")

        count = storage.delete_many(["key1", "key3", "nonexistent"])

        assert count == 2
        assert storage.get("key1") is None
        assert storage.get("key2") == "v2"
        assert storage.get("key3") is None

    def test_size(self):
        """Test getting storage size."""
        storage = MemoryStorage()
        storage.set("key1", "v1")
        storage.set("key2", "v2")

        assert storage.size() == 2

    def test_complex_values(self):
        """Test storing complex values."""
        storage = MemoryStorage()

        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "number": 123.45,
        }

        storage.set("complex", complex_value)

        result = storage.get("complex")
        assert result == complex_value


# =============================================================================
# MemoryDocumentStorage Tests
# =============================================================================

class TestMemoryDocumentStorage:
    """Tests for MemoryDocumentStorage."""

    def test_basic_operations(self):
        """Test basic document operations."""
        storage = MemoryDocumentStorage()

        doc = {"id": "1", "name": "Test", "status": "active"}
        storage.set("doc:1", doc)

        result = storage.get("doc:1")
        assert result == doc

    def test_query_no_filter(self):
        """Test querying without filters."""
        storage = MemoryDocumentStorage()
        storage.set("doc:1", {"id": "1", "status": "active"})
        storage.set("doc:2", {"id": "2", "status": "inactive"})
        storage.set("doc:3", {"id": "3", "status": "active"})

        results = storage.query()

        assert len(results) == 3

    def test_query_with_filter(self):
        """Test querying with filters."""
        storage = MemoryDocumentStorage()
        storage.set("doc:1", {"id": "1", "status": "active"})
        storage.set("doc:2", {"id": "2", "status": "inactive"})
        storage.set("doc:3", {"id": "3", "status": "active"})

        results = storage.query(filters={"status": "active"})

        assert len(results) == 2
        for doc in results:
            assert doc["status"] == "active"

    def test_query_with_list_filter(self):
        """Test querying with list value filter."""
        storage = MemoryDocumentStorage()
        storage.set("doc:1", {"id": "1", "category": "A"})
        storage.set("doc:2", {"id": "2", "category": "B"})
        storage.set("doc:3", {"id": "3", "category": "C"})

        results = storage.query(filters={"category": ["A", "C"]})

        assert len(results) == 2
        categories = [doc["category"] for doc in results]
        assert "A" in categories
        assert "C" in categories

    def test_query_pagination(self):
        """Test query pagination."""
        storage = MemoryDocumentStorage()
        for i in range(10):
            storage.set(f"doc:{i}", {"id": str(i)})

        # Get first page
        page1 = storage.query(limit=3, offset=0)
        assert len(page1) == 3

        # Get second page
        page2 = storage.query(limit=3, offset=3)
        assert len(page2) == 3

        # Ensure different results
        page1_ids = {doc["id"] for doc in page1}
        page2_ids = {doc["id"] for doc in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_count_no_filter(self):
        """Test counting without filters."""
        storage = MemoryDocumentStorage()
        storage.set("doc:1", {"id": "1"})
        storage.set("doc:2", {"id": "2"})
        storage.set("doc:3", {"id": "3"})

        count = storage.count()

        assert count == 3

    def test_count_with_filter(self):
        """Test counting with filters."""
        storage = MemoryDocumentStorage()
        storage.set("doc:1", {"id": "1", "status": "active"})
        storage.set("doc:2", {"id": "2", "status": "inactive"})
        storage.set("doc:3", {"id": "3", "status": "active"})

        count = storage.count(filters={"status": "active"})

        assert count == 2


# =============================================================================
# FileStorage Tests
# =============================================================================

class TestFileStorage:
    """Tests for FileStorage backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        shutil.rmtree(dir_path, ignore_errors=True)

    def test_basic_get_set(self, temp_dir):
        """Test basic get and set operations."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)

        storage.set("key1", "value1")
        assert storage.get("key1") == "value1"

    def test_get_nonexistent_key(self, temp_dir):
        """Test getting a key that doesn't exist."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)

        assert storage.get("nonexistent") is None

    def test_set_overwrites_existing(self, temp_dir):
        """Test that set overwrites existing values."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)

        storage.set("key1", "value1")
        storage.set("key1", "value2")

        assert storage.get("key1") == "value2"

    def test_delete_existing_key(self, temp_dir):
        """Test deleting an existing key."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)
        storage.set("key1", "value1")

        result = storage.delete("key1")

        assert result is True
        assert storage.get("key1") is None

    def test_delete_nonexistent_key(self, temp_dir):
        """Test deleting a key that doesn't exist."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)

        result = storage.delete("nonexistent")

        assert result is False

    def test_exists(self, temp_dir):
        """Test checking if key exists."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)
        storage.set("key1", "value1")

        assert storage.exists("key1") is True
        assert storage.exists("nonexistent") is False

    def test_keys_all(self, temp_dir):
        """Test listing all keys."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)
        storage.set("key1", "v1")
        storage.set("key2", "v2")
        storage.set("key3", "v3")

        keys = storage.keys()

        assert len(keys) == 3
        assert set(keys) == {"key1", "key2", "key3"}

    def test_keys_with_pattern(self, temp_dir):
        """Test listing keys with pattern."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)
        storage.set("user-1", "v1")
        storage.set("user-2", "v2")
        storage.set("order-1", "v3")

        user_keys = storage.keys("user-*")

        assert len(user_keys) == 2
        assert set(user_keys) == {"user-1", "user-2"}

    def test_clear(self, temp_dir):
        """Test clearing all items."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)
        storage.set("key1", "v1")
        storage.set("key2", "v2")

        count = storage.clear()

        assert count == 2
        assert storage.get("key1") is None
        assert storage.get("key2") is None

    def test_namespace_isolation(self, temp_dir):
        """Test namespace isolation."""
        config1 = StorageConfig(path=temp_dir, namespace="ns1")
        config2 = StorageConfig(path=temp_dir, namespace="ns2")

        storage1 = FileStorage(config1)
        storage2 = FileStorage(config2)

        storage1.set("key", "value1")
        storage2.set("key", "value2")

        assert storage1.get("key") == "value1"
        assert storage2.get("key") == "value2"

    def test_ttl_expiration(self, temp_dir):
        """Test TTL expiration."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)
        storage.set("key1", "value1", ttl_seconds=1)

        # Value should exist initially
        assert storage.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Value should be expired
        assert storage.get("key1") is None

    def test_complex_values(self, temp_dir):
        """Test storing complex values."""
        config = StorageConfig(path=temp_dir)
        storage = FileStorage(config)

        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "number": 123.45,
        }

        storage.set("complex", complex_value)

        result = storage.get("complex")
        assert result == complex_value

    def test_persistence(self, temp_dir):
        """Test that data persists across instances."""
        config = StorageConfig(path=temp_dir)

        # Write with first instance
        storage1 = FileStorage(config)
        storage1.set("key1", "value1")

        # Read with second instance
        storage2 = FileStorage(config)
        assert storage2.get("key1") == "value1"


# =============================================================================
# FileDocumentStorage Tests
# =============================================================================

class TestFileDocumentStorage:
    """Tests for FileDocumentStorage."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        shutil.rmtree(dir_path, ignore_errors=True)

    def test_basic_operations(self, temp_dir):
        """Test basic document operations."""
        config = StorageConfig(path=temp_dir)
        storage = FileDocumentStorage(config)

        doc = {"id": "1", "name": "Test", "status": "active"}
        storage.set("doc:1", doc)

        result = storage.get("doc:1")
        assert result == doc

    def test_query_no_filter(self, temp_dir):
        """Test querying without filters."""
        config = StorageConfig(path=temp_dir)
        storage = FileDocumentStorage(config)
        storage.set("doc-1", {"id": "1", "status": "active"})
        storage.set("doc-2", {"id": "2", "status": "inactive"})
        storage.set("doc-3", {"id": "3", "status": "active"})

        results = storage.query()

        assert len(results) == 3

    def test_query_with_filter(self, temp_dir):
        """Test querying with filters."""
        config = StorageConfig(path=temp_dir)
        storage = FileDocumentStorage(config)
        storage.set("doc-1", {"id": "1", "status": "active"})
        storage.set("doc-2", {"id": "2", "status": "inactive"})
        storage.set("doc-3", {"id": "3", "status": "active"})

        results = storage.query(filters={"status": "active"})

        assert len(results) == 2
        for doc in results:
            assert doc["status"] == "active"

    def test_count_no_filter(self, temp_dir):
        """Test counting without filters."""
        config = StorageConfig(path=temp_dir)
        storage = FileDocumentStorage(config)
        storage.set("doc-1", {"id": "1"})
        storage.set("doc-2", {"id": "2"})
        storage.set("doc-3", {"id": "3"})

        count = storage.count()

        assert count == 3

    def test_count_with_filter(self, temp_dir):
        """Test counting with filters."""
        config = StorageConfig(path=temp_dir)
        storage = FileDocumentStorage(config)
        storage.set("doc-1", {"id": "1", "status": "active"})
        storage.set("doc-2", {"id": "2", "status": "inactive"})
        storage.set("doc-3", {"id": "3", "status": "active"})

        count = storage.count(filters={"status": "active"})

        assert count == 2
