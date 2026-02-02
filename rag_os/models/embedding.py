"""Embedding models for RAG OS."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from rag_os.models.chunk import Chunk


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector.

    Attributes:
        chunk: The original chunk
        embedding: The embedding vector (dense)
        sparse_embedding: Optional sparse embedding (for hybrid search)
        model_name: Name of the embedding model used
        model_version: Version of the embedding model
        dimensions: Dimensionality of the embedding
        created_at: When the embedding was created
    """
    chunk: Chunk
    embedding: list[float]
    sparse_embedding: dict[int, float] | None = None
    model_name: str = ""
    model_version: str = "1.0.0"
    dimensions: int = 0
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self):
        """Compute dimensions if not set."""
        if not self.dimensions and self.embedding:
            self.dimensions = len(self.embedding)

    @property
    def chunk_id(self) -> str:
        """Get the chunk ID."""
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        """Get the document ID."""
        return self.chunk.doc_id

    @property
    def content(self) -> str:
        """Get the chunk content."""
        return self.chunk.content

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "embedding": self.embedding,
            "sparse_embedding": self.sparse_embedding,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddedChunk":
        """Deserialize from dictionary."""
        chunk = Chunk.from_dict(data["chunk"])
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = _utc_now()

        return cls(
            chunk=chunk,
            embedding=data["embedding"],
            sparse_embedding=data.get("sparse_embedding"),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", "1.0.0"),
            dimensions=data.get("dimensions", 0),
            created_at=created_at,
        )


class EmbeddingModel(Protocol):
    """Protocol for embedding model implementations.

    Any class implementing this protocol can be used as an embedding model.
    """

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query.

        Some models use different embeddings for queries vs documents.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        ...


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations.

    Attributes:
        model_name: Name of the embedding model
        dimensions: Expected embedding dimensions (0 = auto)
        batch_size: Batch size for embedding
        normalize: Whether to normalize embeddings
        max_retries: Maximum retries for API calls
        timeout: Timeout in seconds
        api_key: API key for cloud providers
        api_base: Base URL for API
    """
    model_name: str = "text-embedding-3-small"
    dimensions: int = 0
    batch_size: int = 100
    normalize: bool = True
    max_retries: int = 3
    timeout: float = 30.0
    api_key: str | None = None
    api_base: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excluding sensitive data)."""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """Deserialize from dictionary."""
        return cls(
            model_name=data.get("model_name", "text-embedding-3-small"),
            dimensions=data.get("dimensions", 0),
            batch_size=data.get("batch_size", 100),
            normalize=data.get("normalize", True),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout", 30.0),
            api_key=data.get("api_key"),
            api_base=data.get("api_base"),
        )


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        embeddings: List of embedding vectors
        model_name: Name of the model used
        dimensions: Embedding dimensions
        token_count: Total tokens processed
        latency_ms: Time taken in milliseconds
    """
    embeddings: list[list[float]]
    model_name: str
    dimensions: int
    token_count: int = 0
    latency_ms: float = 0.0

    @property
    def count(self) -> int:
        """Number of embeddings."""
        return len(self.embeddings)
