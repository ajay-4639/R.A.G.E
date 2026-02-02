"""Base reranking step for RAG OS."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.models.index import SearchResult


@dataclass
class RerankingConfig:
    """Configuration for reranking steps.

    Attributes:
        top_k: Number of results to return after reranking
        min_score: Minimum score threshold
        normalize_scores: Whether to normalize scores to [0, 1]
    """
    top_k: int | None = None  # None = keep all
    min_score: float = 0.0
    normalize_scores: bool = True


class BaseRerankingStep(Step):
    """Abstract base class for reranking steps.

    Reranking steps take search results and reorder them based on
    additional scoring criteria (cross-encoder, LLM, etc.)
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.RERANKING,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)

    def get_config(self) -> RerankingConfig:
        """Get typed configuration."""
        cfg = self.config or {}
        return RerankingConfig(
            top_k=cfg.get("top_k"),
            min_score=cfg.get("min_score", 0.0),
            normalize_scores=cfg.get("normalize_scores", True),
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """Reranking steps accept search results and a query."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["query", "results"],
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Reranking steps return reranked search results."""
        return {
            "type": "array",
            "items": {"type": "object", "description": "SearchResult"},
        }

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank search results based on the query.

        Args:
            query: The original query
            results: Search results to rerank

        Returns:
            Reranked and possibly filtered results
        """
        pass

    def execute(self, context: StepContext) -> StepResult:
        """Execute the reranking step."""
        import time
        start = time.time()

        data = context.data
        config = self.get_config()

        # Parse input
        if isinstance(data, dict):
            query = data.get("query", "")
            raw_results = data.get("results", [])
        else:
            return StepResult.fail(
                "Input must be a dict with 'query' and 'results' keys",
                step_id=self.step_id,
            )

        if not query:
            return StepResult.fail(
                "Query is required",
                step_id=self.step_id,
            )

        # Convert to SearchResult objects
        results: list[SearchResult] = []
        for r in raw_results:
            if isinstance(r, SearchResult):
                results.append(r)
            elif isinstance(r, dict):
                results.append(SearchResult.from_dict(r))
            else:
                return StepResult.fail(
                    f"Invalid result type: {type(r)}",
                    step_id=self.step_id,
                )

        try:
            reranked = self.rerank(query, results)

            # Apply top_k
            if config.top_k:
                reranked = reranked[:config.top_k]

            # Apply min_score
            if config.min_score > 0:
                reranked = [r for r in reranked if r.score >= config.min_score]

            latency = (time.time() - start) * 1000

            return StepResult.ok(
                output=reranked,
                result_count=len(reranked),
                original_count=len(results),
            ).with_latency(latency)

        except Exception as e:
            return StepResult.fail(
                str(e),
                step_id=self.step_id,
            )

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            for r in results:
                r.score = 1.0
        else:
            for r in results:
                r.score = (r.score - min_score) / (max_score - min_score)

        return results
