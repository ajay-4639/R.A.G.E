"""Chunk model for RAG OS."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import hashlib
import json


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class Chunk:
    """
    A chunk of text derived from a document.

    Chunks are the atomic units used for embedding and retrieval
    in RAG pipelines. Each chunk belongs to a parent document.

    Attributes:
        chunk_id: Unique identifier for this chunk
        doc_id: ID of the parent document
        content: The text content of the chunk
        index: Position of this chunk in the document (0-indexed)
        metadata: Additional metadata
        token_count: Number of tokens (if computed)
        char_count: Number of characters
        start_char: Starting character position in source document
        end_char: Ending character position in source document
        parent_chunk_id: ID of parent chunk (for hierarchical chunking)
        child_chunk_ids: IDs of child chunks (for hierarchical chunking)
        embedding: Vector embedding (populated by embedding step)
        created_at: When this chunk was created
    """

    content: str
    doc_id: str
    index: int = 0
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int | None = None
    char_count: int = 0
    start_char: int = 0
    end_char: int = 0
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Compute char_count if not provided."""
        if self.char_count == 0 and self.content:
            self.char_count = len(self.content)

    @property
    def content_hash(self) -> str:
        """Compute hash of content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def is_embedded(self) -> bool:
        """Check if this chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0

    @property
    def has_children(self) -> bool:
        """Check if this chunk has child chunks."""
        return len(self.child_chunk_ids) > 0

    @property
    def has_parent(self) -> bool:
        """Check if this chunk has a parent chunk."""
        return self.parent_chunk_id is not None

    def set_embedding(self, embedding: list[float]) -> None:
        """Set the embedding vector."""
        self.embedding = embedding

    def add_child(self, child_chunk_id: str) -> None:
        """Add a child chunk ID."""
        if child_chunk_id not in self.child_chunk_ids:
            self.child_chunk_ids.append(child_chunk_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "index": self.index,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            chunk_id=data.get("chunk_id", str(uuid4())),
            doc_id=data["doc_id"],
            content=data["content"],
            index=data.get("index", 0),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
            char_count=data.get("char_count", len(data["content"])),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            parent_chunk_id=data.get("parent_chunk_id"),
            child_chunk_ids=data.get("child_chunk_ids", []),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else _utc_now(),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"Chunk(chunk_id='{self.chunk_id[:8]}...', "
            f"doc_id='{self.doc_id[:8]}...', "
            f"index={self.index}, "
            f"char_count={self.char_count})"
        )


@dataclass
class ChunkWithScore:
    """A chunk with an associated relevance score (used in retrieval)."""

    chunk: Chunk
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "ChunkWithScore") -> bool:
        """Compare by score for sorting."""
        return self.score < other.score

    def __repr__(self) -> str:
        return f"ChunkWithScore(chunk_id='{self.chunk.chunk_id[:8]}...', score={self.score:.4f})"


def create_chunk(
    content: str,
    doc_id: str,
    index: int = 0,
    start_char: int = 0,
    end_char: int | None = None,
    parent_chunk_id: str | None = None,
    **metadata: Any,
) -> Chunk:
    """
    Factory function to create a chunk with common defaults.

    Args:
        content: Chunk content
        doc_id: Parent document ID
        index: Position in document
        start_char: Starting character position
        end_char: Ending character position (defaults to start + len)
        parent_chunk_id: Optional parent chunk ID
        **metadata: Additional metadata

    Returns:
        New Chunk instance
    """
    if end_char is None:
        end_char = start_char + len(content)

    return Chunk(
        content=content,
        doc_id=doc_id,
        index=index,
        start_char=start_char,
        end_char=end_char,
        parent_chunk_id=parent_chunk_id,
        metadata=metadata,
    )
