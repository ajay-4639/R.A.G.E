"""File-based storage implementation."""

import json
import os
import fnmatch
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from threading import Lock

from rag_os.storage.base import BaseStorage, BaseDocumentStorage, StorageConfig


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class FileStorage(BaseStorage[Any]):
    """File-based storage backend using JSON files.

    Each key is stored as a separate JSON file.
    Good for simple persistence without external dependencies.
    """

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._lock = Lock()
        self._base_path = Path(config.path if config and config.path else ".rag_os_storage")
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        namespace_path = self._base_path / self.config.namespace
        namespace_path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self._base_path / self.config.namespace / f"{safe_key}.json"

    def get(self, key: str) -> Any | None:
        """Get an item by key."""
        with self._lock:
            path = self._key_to_path(key)

            if not path.exists():
                return None

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check expiration
                expires_at = data.get("_expires_at")
                if expires_at:
                    expires = datetime.fromisoformat(expires_at)
                    if _utc_now() > expires:
                        path.unlink()
                        return None

                return data.get("value")

            except (json.JSONDecodeError, IOError):
                return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store an item."""
        with self._lock:
            path = self._key_to_path(key)
            ttl = ttl_seconds or self.config.ttl_seconds

            data: dict[str, Any] = {
                "key": key,
                "value": value,
                "_created_at": _utc_now().isoformat(),
                "_updated_at": _utc_now().isoformat(),
            }

            if ttl:
                expires = _utc_now() + timedelta(seconds=ttl)
                data["_expires_at"] = expires.isoformat()

            # Load existing creation time
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                        data["_created_at"] = existing.get("_created_at", data["_created_at"])
                except (json.JSONDecodeError, IOError):
                    pass

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    def delete(self, key: str) -> bool:
        """Delete an item."""
        with self._lock:
            path = self._key_to_path(key)

            if path.exists():
                path.unlink()
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if an item exists."""
        # Use get to also check expiration
        return self.get(key) is not None

    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching a pattern."""
        with self._lock:
            namespace_path = self._base_path / self.config.namespace
            if not namespace_path.exists():
                return []

            matching = []
            for file in namespace_path.glob("*.json"):
                key = file.stem.replace("__", "/")
                if fnmatch.fnmatch(key, pattern):
                    # Check if expired
                    if self.get(key) is not None:
                        matching.append(key)

            return matching

    def clear(self) -> int:
        """Clear all items in namespace."""
        with self._lock:
            namespace_path = self._base_path / self.config.namespace
            if not namespace_path.exists():
                return 0

            count = 0
            for file in namespace_path.glob("*.json"):
                file.unlink()
                count += 1

            return count


class FileDocumentStorage(BaseDocumentStorage):
    """File-based document storage with query support."""

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._storage = FileStorage(config)

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

        return results[offset:offset + limit]

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count documents matching filters."""
        if not filters:
            return len(self._storage.keys("*"))
        return len(self.query(filters, limit=100000))

    def _matches_filters(self, doc: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if document matches all filters."""
        for key, value in filters.items():
            doc_value = doc.get(key)

            if isinstance(value, list):
                if doc_value not in value:
                    return False
            elif doc_value != value:
                return False

        return True
