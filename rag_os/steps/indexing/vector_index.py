"""Vector index implementations for RAG OS."""

import math
from datetime import datetime, timezone
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.embedding import EmbeddedChunk
from rag_os.models.index import (
    SearchResult,
    IndexStats,
    IndexConfig,
    DistanceMetric,
    IndexType,
)
from rag_os.steps.indexing.base import BaseIndex, BaseIndexingStep


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class InMemoryVectorIndex(BaseIndex):
    """Simple in-memory vector index.

    Uses brute-force search for simplicity. Good for small datasets
    and testing. For production, use FAISS or a vector database.
    """

    def __init__(self, config: IndexConfig | None = None):
        super().__init__(config)
        self._chunks: dict[str, EmbeddedChunk] = {}
        self._doc_ids: set[str] = set()
        self._created_at = _utc_now()
        self._updated_at = _utc_now()

    def add(self, chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the index."""
        count = 0
        for chunk in chunks:
            if chunk.chunk_id not in self._chunks:
                self._chunks[chunk.chunk_id] = chunk
                self._doc_ids.add(chunk.doc_id)
                count += 1
            else:
                # Update existing
                self._chunks[chunk.chunk_id] = chunk
                count += 1

        self._updated_at = _utc_now()
        return count

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the index with brute-force similarity search."""
        results: list[tuple[float, EmbeddedChunk]] = []

        for chunk in self._chunks.values():
            # Apply metadata filters
            if filters:
                if not self._matches_filters(chunk, filters):
                    continue

            # Calculate similarity
            score = self._calculate_similarity(
                query_embedding,
                chunk.embedding,
                self.config.metric,
            )
            results.append((score, chunk))

        # Sort by score (descending for similarity)
        results.sort(key=lambda x: x[0], reverse=True)

        # Return top_k
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                score=score,
                metadata=chunk.chunk.metadata,
                embedding=chunk.embedding,
            )
            for score, chunk in results[:top_k]
        ]

    def _matches_filters(self, chunk: EmbeddedChunk, filters: dict[str, Any]) -> bool:
        """Check if chunk matches all filters."""
        metadata = chunk.chunk.metadata

        for key, value in filters.items():
            # Handle special filters
            if key == "doc_id":
                if chunk.doc_id != value:
                    return False
            elif key == "doc_ids":
                if chunk.doc_id not in value:
                    return False
            elif key not in metadata:
                return False
            elif isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False

        return True

    def _calculate_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
        metric: DistanceMetric,
    ) -> float:
        """Calculate similarity between two vectors."""
        if metric == DistanceMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif metric == DistanceMetric.EUCLIDEAN:
            # Convert distance to similarity
            dist = self._euclidean_distance(vec1, vec2)
            return 1 / (1 + dist)
        elif metric == DistanceMetric.DOT_PRODUCT:
            return self._dot_product(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _euclidean_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    def _dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate dot product."""
        return sum(a * b for a, b in zip(vec1, vec2))

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks from the index."""
        count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                count += 1

        # Rebuild doc_ids set
        self._doc_ids = {chunk.doc_id for chunk in self._chunks.values()}
        self._updated_at = _utc_now()
        return count

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        # Estimate size (rough approximation)
        size = 0
        dims = 0
        for chunk in self._chunks.values():
            if chunk.embedding:
                dims = len(chunk.embedding)
                size += dims * 4  # 4 bytes per float
            size += len(chunk.content)

        return IndexStats(
            total_chunks=len(self._chunks),
            total_documents=len(self._doc_ids),
            index_size_bytes=size,
            dimensions=dims,
            index_type=IndexType.DENSE,
            metric=self.config.metric,
            created_at=self._created_at,
            updated_at=self._updated_at,
        )

    def clear(self) -> None:
        """Clear all data from the index."""
        self._chunks.clear()
        self._doc_ids.clear()
        self._updated_at = _utc_now()

    def get_chunk(self, chunk_id: str) -> EmbeddedChunk | None:
        """Get a chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> list[EmbeddedChunk]:
        """Get all chunks in the index."""
        return list(self._chunks.values())


@register_step(
    name="IndexingStep",
    step_type=StepType.INDEXING,
    description="Add embedded chunks to a vector index",
    version="1.0.0",
)
class IndexingStep(BaseIndexingStep):
    """Standard indexing step that adds embedded chunks to an index.

    Can work with any BaseIndex implementation.
    """

    def __init__(
        self,
        step_id: str,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
    ):
        # Create default in-memory index if none provided
        if index is None:
            index_config = IndexConfig(
                dimensions=config.get("dimensions", 384) if config else 384,
                metric=DistanceMetric(config.get("metric", "cosine")) if config else DistanceMetric.COSINE,
            )
            index = InMemoryVectorIndex(index_config)

        super().__init__(step_id, config=config, index=index)

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """Add or update chunks in the index.

        Args:
            chunks: Chunks to upsert

        Returns:
            Number of chunks upserted
        """
        return self.index.add(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the index.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Optional filters

        Returns:
            Search results
        """
        return self.index.search(query_embedding, top_k, filters)
