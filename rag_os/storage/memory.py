"""In-memory storage implementation."""

import fnmatch
from datetime import datetime, timezone, timedelta
from typing import Any
from threading import Lock

from rag_os.storage.base import BaseStorage, BaseDocumentStorage, StorageConfig, StoredItem


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class MemoryStorage(BaseStorage[Any]):
    """Thread-safe in-memory storage backend.

    Good for development, testing, and small-scale deployments.
    Data is lost when the process exits.
    """

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._data: dict[str, StoredItem[Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        """Get an item by key."""
        with self._lock:
            prefixed_key = self._prefix_key(key)
            item = self._data.get(prefixed_key)

            if item is None:
                return None

            if item.is_expired:
                del self._data[prefixed_key]
                return None

            return item.value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store an item."""
        with self._lock:
            prefixed_key = self._prefix_key(key)
            ttl = ttl_seconds or self.config.ttl_seconds

            expires_at = None
            if ttl:
                expires_at = _utc_now() + timedelta(seconds=ttl)

            now = _utc_now()
            existing = self._data.get(prefixed_key)

            self._data[prefixed_key] = StoredItem(
                key=key,
                value=value,
                created_at=existing.created_at if existing else now,
                updated_at=now,
                expires_at=expires_at,
            )

    def delete(self, key: str) -> bool:
        """Delete an item."""
        with self._lock:
            prefixed_key = self._prefix_key(key)
            if prefixed_key in self._data:
                del self._data[prefixed_key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if an item exists."""
        with self._lock:
            prefixed_key = self._prefix_key(key)
            item = self._data.get(prefixed_key)

            if item is None:
                return False

            if item.is_expired:
                del self._data[prefixed_key]
                return False

            return True

    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching a pattern."""
        with self._lock:
            self._cleanup_expired()
            prefix = f"{self.config.namespace}:" if self.config.namespace else ""
            full_pattern = f"{prefix}{pattern}"

            matching = []
            for key in self._data.keys():
                if fnmatch.fnmatch(key, full_pattern):
                    # Remove prefix for return
                    clean_key = key[len(prefix):] if prefix and key.startswith(prefix) else key
                    matching.append(clean_key)

            return matching

    def clear(self) -> int:
        """Clear all items in namespace."""
        with self._lock:
            prefix = f"{self.config.namespace}:" if self.config.namespace else ""
            to_delete = [k for k in self._data.keys() if k.startswith(prefix)]
            for key in to_delete:
                del self._data[key]
            return len(to_delete)

    def _prefix_key(self, key: str) -> str:
        """Add namespace prefix to key."""
        if self.config.namespace:
            return f"{self.config.namespace}:{key}"
        return key

    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        expired = [k for k, v in self._data.items() if v.is_expired]
        for key in expired:
            del self._data[key]

    def size(self) -> int:
        """Get number of items."""
        with self._lock:
            self._cleanup_expired()
            prefix = f"{self.config.namespace}:" if self.config.namespace else ""
            return sum(1 for k in self._data.keys() if k.startswith(prefix))


class MemoryDocumentStorage(BaseDocumentStorage):
    """In-memory document storage with query support."""

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._storage = MemoryStorage(config)

    def get(self, key: str) -> dict[str, Any] | None:
        """Get a document by key."""
        return self._storage.get(key)

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        """Store a document."""
        self._storage.set(key, value, ttl_seconds)

    def delete(self, key: str) -> bool:
        """Delete a document."""
        return self._storage.delete(key)

    def exists(self, key: str) -> bool:
        """Check if a document exists."""
        return self._storage.exists(key)

    def keys(self, pattern: str = "*") -> list[str]:
        """List document keys matching pattern."""
        return self._storage.keys(pattern)

    def clear(self) -> int:
        """Clear all documents."""
        return self._storage.clear()

    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query documents with filters."""
        all_keys = self._storage.keys("*")
        results = []

        for key in all_keys:
            doc = self._storage.get(key)
            if doc is None:
                continue

            if filters and not self._matches_filters(doc, filters):
                continue

            results.append(doc)

        # Apply pagination
        return results[offset:offset + limit]

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count documents matching filters."""
        if not filters:
            return self._storage.size()

        return len(self.query(filters, limit=100000))

    def _matches_filters(self, doc: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if document matches all filters."""
        for key, value in filters.items():
            doc_value = doc.get(key)

            if isinstance(value, list):
                # Value in list
                if doc_value not in value:
                    return False
            elif doc_value != value:
                return False

        return True
