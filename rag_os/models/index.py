"""Index and search models for RAG OS."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from enum import Enum


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class IndexType(Enum):
    """Type of index."""
    DENSE = "dense"         # Dense vector index (embeddings)
    SPARSE = "sparse"       # Sparse vector index (BM25, TF-IDF)
    HYBRID = "hybrid"       # Combination of dense and sparse


class DistanceMetric(Enum):
    """Distance/similarity metric for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        chunk_id: ID of the matching chunk
        doc_id: ID of the parent document
        content: The chunk content
        score: Relevance score (higher = more relevant)
        metadata: Additional metadata
        embedding: Optional embedding vector
    """
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            content=data["content"],
            score=data["score"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class SearchQuery:
    """A search query with parameters.

    Attributes:
        query_text: The query text
        query_embedding: Optional pre-computed embedding
        top_k: Number of results to return
        filters: Metadata filters
        min_score: Minimum score threshold
        include_metadata: Whether to include metadata in results
        include_embeddings: Whether to include embeddings in results
    """
    query_text: str
    query_embedding: list[float] | None = None
    top_k: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0
    include_metadata: bool = True
    include_embeddings: bool = False


@dataclass
class SearchResponse:
    """Response from a search operation.

    Attributes:
        results: List of search results
        query: The original query
        total_count: Total number of matches (before top_k limit)
        latency_ms: Search latency in milliseconds
        metadata: Additional response metadata
    """
    results: list[SearchResult]
    query: SearchQuery
    total_count: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.results)

    def top(self, n: int = 1) -> list[SearchResult]:
        """Get top N results."""
        return self.results[:n]


@dataclass
class IndexStats:
    """Statistics about an index.

    Attributes:
        total_chunks: Total number of chunks indexed
        total_documents: Number of unique documents
        index_size_bytes: Size of the index in bytes
        dimensions: Embedding dimensions
        index_type: Type of index
        metric: Distance metric used
        created_at: When the index was created
        updated_at: When the index was last updated
    """
    total_chunks: int = 0
    total_documents: int = 0
    index_size_bytes: int = 0
    dimensions: int = 0
    index_type: IndexType = IndexType.DENSE
    metric: DistanceMetric = DistanceMetric.COSINE
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "total_documents": self.total_documents,
            "index_size_bytes": self.index_size_bytes,
            "dimensions": self.dimensions,
            "index_type": self.index_type.value,
            "metric": self.metric.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class IndexConfig:
    """Configuration for an index.

    Attributes:
        name: Index name
        dimensions: Embedding dimensions
        metric: Distance metric
        index_type: Type of index
        namespace: Optional namespace for multi-tenancy
        ef_construction: HNSW construction parameter
        ef_search: HNSW search parameter
        m: HNSW M parameter
    """
    name: str = "default"
    dimensions: int = 384
    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.DENSE
    namespace: str | None = None
    ef_construction: int = 100
    ef_search: int = 50
    m: int = 16

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "dimensions": self.dimensions,
            "metric": self.metric.value,
            "index_type": self.index_type.value,
            "namespace": self.namespace,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "m": self.m,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexConfig":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", "default"),
            dimensions=data.get("dimensions", 384),
            metric=DistanceMetric(data.get("metric", "cosine")),
            index_type=IndexType(data.get("index_type", "dense")),
            namespace=data.get("namespace"),
            ef_construction=data.get("ef_construction", 100),
            ef_search=data.get("ef_search", 50),
            m=data.get("m", 16),
        )
