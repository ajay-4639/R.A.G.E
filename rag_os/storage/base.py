"""Base storage interfaces for RAG OS."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar, Generic
from enum import Enum


T = TypeVar('T')


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class StorageType(Enum):
    """Types of storage backends."""
    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    REDIS = "redis"
    S3 = "s3"


@dataclass
class StorageConfig:
    """Configuration for storage backends.

    Attributes:
        storage_type: Type of storage backend
        path: Path for file-based storage
        connection_string: Connection string for databases
        namespace: Namespace for multi-tenancy
        ttl_seconds: Time-to-live for cached items
    """
    storage_type: StorageType = StorageType.MEMORY
    path: str = ""
    connection_string: str = ""
    namespace: str = "default"
    ttl_seconds: int | None = None


@dataclass
class StoredItem(Generic[T]):
    """A stored item with metadata.

    Attributes:
        key: Unique key for the item
        value: The stored value
        created_at: When the item was created
        updated_at: When the item was last updated
        expires_at: When the item expires (optional)
        metadata: Additional metadata
    """
    key: str
    value: T
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the item has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class BaseStorage(ABC, Generic[T]):
    """Abstract base class for storage backends.

    Provides a simple key-value interface for storing and retrieving data.
    """

    def __init__(self, config: StorageConfig | None = None):
        """Initialize storage with configuration."""
        self.config = config or StorageConfig()

    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get an item by key.

        Args:
            key: The item key

        Returns:
            The stored value or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl_seconds: int | None = None) -> None:
        """Store an item.

        Args:
            key: The item key
            value: The value to store
            ttl_seconds: Optional time-to-live in seconds
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an item.

        Args:
            key: The item key

        Returns:
            True if item was deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an item exists.

        Args:
            key: The item key

        Returns:
            True if item exists
        """
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching a pattern.

        Args:
            pattern: Glob-style pattern (e.g., "user:*")

        Returns:
            List of matching keys
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all items.

        Returns:
            Number of items cleared
        """
        pass

    def get_many(self, keys: list[str]) -> dict[str, T]:
        """Get multiple items by keys.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict mapping keys to values (missing keys omitted)
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def set_many(self, items: dict[str, T], ttl_seconds: int | None = None) -> None:
        """Store multiple items.

        Args:
            items: Dict mapping keys to values
            ttl_seconds: Optional time-to-live for all items
        """
        for key, value in items.items():
            self.set(key, value, ttl_seconds)

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple items.

        Args:
            keys: List of keys to delete

        Returns:
            Number of items deleted
        """
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count


class BaseDocumentStorage(BaseStorage[dict[str, Any]]):
    """Storage specialized for documents with indexing support."""

    @abstractmethod
    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query documents with filters.

        Args:
            filters: Field filters (e.g., {"status": "active"})
            limit: Maximum results
            offset: Skip first N results

        Returns:
            List of matching documents
        """
        pass

    @abstractmethod
    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count documents matching filters.

        Args:
            filters: Field filters

        Returns:
            Count of matching documents
        """
        pass
