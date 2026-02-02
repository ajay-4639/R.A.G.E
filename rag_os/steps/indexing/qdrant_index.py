"""Qdrant vector index implementation for RAG OS."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from rag_os.models.embedding import EmbeddedChunk
from rag_os.models.index import (
    SearchResult,
    IndexStats,
    IndexConfig,
    DistanceMetric,
    IndexType,
)
from rag_os.steps.indexing.base import BaseIndex

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        FilterSelector,
    )
except ImportError:
    QdrantClient = None  # type: ignore


def _distance_from_metric(metric: DistanceMetric) -> "Distance":
    """Convert RAG OS DistanceMetric to Qdrant Distance."""
    mapping = {
        DistanceMetric.COSINE: Distance.COSINE,
        DistanceMetric.EUCLIDEAN: Distance.EUCLID,
        DistanceMetric.DOT_PRODUCT: Distance.DOT,
    }
    return mapping.get(metric, Distance.COSINE)


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Ensure chunk_id is a valid UUID string for Qdrant point IDs."""
    try:
        UUID(chunk_id)
        return chunk_id
    except ValueError:
        # Generate a deterministic UUID from the chunk_id string
        import hashlib
        hash_bytes = hashlib.md5(chunk_id.encode()).hexdigest()
        return f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-{hash_bytes[16:20]}-{hash_bytes[20:32]}"


class QdrantVectorIndex(BaseIndex):
    """Vector index backed by Qdrant.

    Connects to a Qdrant instance (local or remote) and provides
    the standard BaseIndex interface for adding, searching, and
    deleting embedded chunks.

    Usage:
        from rag_os.models.index import IndexConfig, DistanceMetric

        config = IndexConfig(dimensions=1536, metric=DistanceMetric.COSINE)
        index = QdrantVectorIndex(
            host="localhost",
            port=6333,
            collection_name="my_docs",
            config=config,
        )

        # Add chunks
        index.add(embedded_chunks)

        # Search
        results = index.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rag_os_default",
        config: IndexConfig | None = None,
    ):
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client is required for QdrantVectorIndex. "
                "Install it with: pip install qdrant-client"
            )

        super().__init__(config)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = QdrantClient(host=host, port=port)
        self._created_at = datetime.now(timezone.utc)

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimensions or 1536,
                    distance=_distance_from_metric(self.config.metric),
                ),
            )

    def add(self, chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the Qdrant collection."""
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            point_id = _chunk_id_to_uuid(chunk.chunk_id)
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "metadata": chunk.chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the Qdrant collection with a query embedding."""
        # Build Qdrant filter
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    doc_id=payload.get("doc_id", ""),
                    content=payload.get("content", ""),
                    score=hit.score,
                    metadata=payload.get("metadata", {}),
                )
            )

        return search_results

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """Convert a dict of filters to a Qdrant Filter."""
        conditions = []
        for key, value in filters.items():
            if key == "doc_id":
                conditions.append(
                    FieldCondition(key="doc_id", match=MatchValue(value=value))
                )
            elif key == "doc_ids":
                # Multiple doc_ids - use individual conditions with should
                doc_conditions = [
                    FieldCondition(key="doc_id", match=MatchValue(value=did))
                    for did in value
                ]
                return Filter(
                    must=conditions if conditions else None,
                    should=doc_conditions,
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=conditions) if conditions else Filter()

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by their chunk IDs."""
        if not chunk_ids:
            return 0

        point_ids = [_chunk_id_to_uuid(cid) for cid in chunk_ids]
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )
        return len(chunk_ids)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: The document ID whose chunks should be deleted

        Returns:
            Number of points deleted (estimated)
        """
        # First count how many will be deleted
        count_before = self._client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        ).count

        # Delete by filter
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )

        return count_before

    def get_stats(self) -> IndexStats:
        """Get statistics about the Qdrant collection."""
        try:
            info = self._client.get_collection(self.collection_name)
            return IndexStats(
                total_chunks=info.points_count or 0,
                total_documents=0,  # Qdrant doesn't track unique doc_ids
                index_size_bytes=0,
                dimensions=info.config.params.vectors.size if info.config.params.vectors else 0,
                index_type=IndexType.DENSE,
                metric=self.config.metric,
                created_at=self._created_at,
                updated_at=datetime.now(timezone.utc),
            )
        except Exception:
            return IndexStats(
                total_chunks=0,
                total_documents=0,
                index_size_bytes=0,
                dimensions=self.config.dimensions,
                index_type=IndexType.DENSE,
                metric=self.config.metric,
                created_at=self._created_at,
                updated_at=datetime.now(timezone.utc),
            )

    def clear(self) -> None:
        """Clear all data by recreating the collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()

    def count(self) -> int:
        """Get the total number of points in the collection."""
        try:
            return self._client.count(collection_name=self.collection_name).count
        except Exception:
            return 0
