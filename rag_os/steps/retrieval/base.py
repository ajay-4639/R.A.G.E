"""Base retrieval step for RAG OS."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.models.index import SearchResult, SearchQuery, SearchResponse
from rag_os.steps.indexing.base import BaseIndex
from rag_os.steps.embedding.base import BaseEmbeddingStep


@dataclass
class RetrievalConfig:
    """Configuration for retrieval steps.

    Attributes:
        top_k: Number of results to retrieve
        min_score: Minimum relevance score threshold
        rerank: Whether to apply reranking
        include_metadata: Include metadata in results
        filters: Default metadata filters to apply
    """
    top_k: int = 10
    min_score: float = 0.0
    rerank: bool = False
    include_metadata: bool = True
    filters: dict[str, Any] | None = None


class BaseRetrievalStep(Step):
    """Abstract base class for retrieval steps.

    Retrieval steps take a query and return relevant chunks from an index.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.RETRIEVAL,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
        embedder: BaseEmbeddingStep | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)
        self._index = index
        self._embedder = embedder

    @property
    def index(self) -> BaseIndex:
        """Get the index. Must be set before use."""
        if self._index is None:
            raise ValueError("Index not set. Call set_index() first.")
        return self._index

    def set_index(self, index: BaseIndex) -> None:
        """Set the index to use."""
        self._index = index

    @property
    def embedder(self) -> BaseEmbeddingStep:
        """Get the embedder. Must be set before use."""
        if self._embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        return self._embedder

    def set_embedder(self, embedder: BaseEmbeddingStep) -> None:
        """Set the embedding step to use for query embedding."""
        self._embedder = embedder

    def get_config(self) -> RetrievalConfig:
        """Get typed configuration."""
        cfg = self.config or {}
        return RetrievalConfig(
            top_k=cfg.get("top_k", 10),
            min_score=cfg.get("min_score", 0.0),
            rerank=cfg.get("rerank", False),
            include_metadata=cfg.get("include_metadata", True),
            filters=cfg.get("filters"),
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """Retrieval steps accept a query string or SearchQuery."""
        return {
            "anyOf": [
                {"type": "string", "description": "Query text"},
                {"type": "object", "description": "SearchQuery object"},
            ]
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Retrieval steps return List[SearchResult]."""
        return {
            "type": "array",
            "items": {"type": "object", "description": "SearchResult"},
        }

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The query text
            top_k: Number of results (overrides config)
            filters: Metadata filters (merged with config filters)

        Returns:
            List of search results sorted by relevance
        """
        pass

    def execute(self, context: StepContext) -> StepResult:
        """Execute the retrieval step.

        Accepts a query string or SearchQuery object.
        """
        import time
        start = time.time()

        data = context.data
        config = self.get_config()

        # Parse input
        if isinstance(data, str):
            query_text = data
            top_k = config.top_k
            filters = config.filters
        elif isinstance(data, dict):
            query_text = data.get("query_text", data.get("query", ""))
            top_k = data.get("top_k", config.top_k)
            filters = data.get("filters", config.filters)
        elif isinstance(data, SearchQuery):
            query_text = data.query_text
            top_k = data.top_k
            filters = data.filters
        else:
            return StepResult.fail(
                f"Invalid input type: {type(data)}",
                step_id=self.step_id,
            )

        if not query_text:
            return StepResult.fail(
                "Query text is required",
                step_id=self.step_id,
            )

        try:
            results = self.retrieve(query_text, top_k, filters)

            # Apply min_score filter
            if config.min_score > 0:
                results = [r for r in results if r.score >= config.min_score]

            latency = (time.time() - start) * 1000

            return StepResult.ok(
                output=results,
                result_count=len(results),
                query=query_text,
            ).with_latency(latency)

        except Exception as e:
            return StepResult.fail(
                str(e),
                step_id=self.step_id,
            )
