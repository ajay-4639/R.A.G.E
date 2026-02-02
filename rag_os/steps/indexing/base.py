"""Base index interface and indexing step."""

from abc import ABC, abstractmethod
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.models.embedding import EmbeddedChunk
from rag_os.models.index import (
    SearchResult,
    SearchQuery,
    SearchResponse,
    IndexStats,
    IndexConfig,
    DistanceMetric,
    IndexType,
)


class BaseIndex(ABC):
    """Abstract base class for all index implementations.

    An index stores embedded chunks and provides search functionality.
    Implementations can use various backends (in-memory, Chroma, Pinecone, etc.)
    """

    def __init__(self, config: IndexConfig | None = None):
        """Initialize the index.

        Args:
            config: Index configuration
        """
        self.config = config or IndexConfig()

    @abstractmethod
    def add(self, chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the index.

        Args:
            chunks: List of embedded chunks to add

        Returns:
            Number of chunks added
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the index with a query embedding.

        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results sorted by relevance
        """
        pass

    def search_with_query(self, query: SearchQuery) -> SearchResponse:
        """Search the index with a SearchQuery object.

        Args:
            query: The search query

        Returns:
            SearchResponse with results and metadata
        """
        import time
        start = time.time()

        if query.query_embedding is None:
            raise ValueError("Query embedding is required")

        results = self.search(
            query_embedding=query.query_embedding,
            top_k=query.top_k,
            filters=query.filters if query.filters else None,
        )

        # Apply min_score filter
        if query.min_score > 0:
            results = [r for r in results if r.score >= query.min_score]

        # Filter fields based on query settings
        if not query.include_embeddings:
            for r in results:
                r.embedding = None

        latency = (time.time() - start) * 1000

        return SearchResponse(
            results=results,
            query=query,
            total_count=len(results),
            latency_ms=latency,
        )

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks from the index.

        Args:
            chunk_ids: IDs of chunks to delete

        Returns:
            Number of chunks deleted
        """
        pass

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Get statistics about the index.

        Returns:
            IndexStats with count, size, etc.
        """
        pass

    def clear(self) -> None:
        """Clear all data from the index."""
        pass

    def persist(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: Path to save the index
        """
        raise NotImplementedError("This index does not support persistence")

    @classmethod
    def load(cls, path: str) -> "BaseIndex":
        """Load an index from disk.

        Args:
            path: Path to load the index from

        Returns:
            Loaded index instance
        """
        raise NotImplementedError("This index does not support persistence")


class BaseIndexingStep(Step):
    """Base class for indexing pipeline steps.

    Indexing steps take embedded chunks and add them to an index.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.INDEXING,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)
        self._index = index

    @property
    def index(self) -> BaseIndex:
        """Get the index. Must be set before use."""
        if self._index is None:
            raise ValueError("Index not set. Call set_index() first.")
        return self._index

    def set_index(self, index: BaseIndex) -> None:
        """Set the index to use.

        Args:
            index: The index instance
        """
        self._index = index

    @property
    def input_schema(self) -> dict[str, Any]:
        """Indexing steps accept List[EmbeddedChunk]."""
        return {
            "type": "array",
            "items": {"type": "object", "description": "EmbeddedChunk"},
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Indexing steps return indexing statistics."""
        return {
            "type": "object",
            "properties": {
                "chunks_indexed": {"type": "integer"},
                "index_stats": {"type": "object"},
            },
        }

    def execute(self, context: StepContext) -> StepResult:
        """Execute the indexing step.

        Args:
            context: Context with embedded chunks

        Returns:
            StepResult with indexing statistics
        """
        import time
        start = time.time()

        data = context.data
        if not data:
            return StepResult.ok(
                output={"chunks_indexed": 0},
                chunks_indexed=0,
            )

        # Ensure we have a list
        if not isinstance(data, list):
            data = [data]

        # Convert to EmbeddedChunk if needed
        chunks: list[EmbeddedChunk] = []
        for item in data:
            if isinstance(item, EmbeddedChunk):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(EmbeddedChunk.from_dict(item))
            else:
                return StepResult.fail(
                    f"Invalid input type: {type(item)}",
                    step_id=self.step_id,
                )

        try:
            count = self.index.add(chunks)
            latency = (time.time() - start) * 1000

            stats = self.index.get_stats()

            return StepResult.ok(
                output={"chunks_indexed": count, "index_stats": stats.to_dict()},
                chunks_indexed=count,
                total_in_index=stats.total_chunks,
            ).with_latency(latency)

        except Exception as e:
            return StepResult.fail(
                str(e),
                step_id=self.step_id,
            )
