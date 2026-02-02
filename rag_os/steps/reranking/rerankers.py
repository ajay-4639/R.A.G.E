"""Reranking implementations for RAG OS."""

from typing import Any
from datetime import datetime, timezone

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.index import SearchResult
from rag_os.steps.reranking.base import BaseRerankingStep


@register_step(
    name="CrossEncoderRerankingStep",
    step_type=StepType.RERANKING,
    description="Rerank using cross-encoder models",
    version="1.0.0",
)
class CrossEncoderRerankingStep(BaseRerankingStep):
    """Reranking using cross-encoder models.

    Cross-encoders score query-document pairs jointly for
    more accurate relevance scoring.
    """

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._model = None
        self._model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2") if config else "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _get_model(self):
        """Load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )

            self._model = CrossEncoder(self._model_name)

        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank using cross-encoder scores."""
        if not results:
            return results

        model = self._get_model()
        config = self.get_config()

        # Create query-document pairs
        pairs = [(query, r.content) for r in results]

        # Get cross-encoder scores
        scores = model.predict(pairs)

        # Update scores
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by new scores
        results.sort(key=lambda r: r.score, reverse=True)

        if config.normalize_scores:
            results = self._normalize_scores(results)

        return results


@register_step(
    name="CohereRerankingStep",
    step_type=StepType.RERANKING,
    description="Rerank using Cohere rerank API",
    version="1.0.0",
)
class CohereRerankingStep(BaseRerankingStep):
    """Reranking using Cohere's rerank API."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._client = None
        self._model = config.get("model", "rerank-english-v3.0") if config else "rerank-english-v3.0"

    def _get_client(self):
        """Get Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "cohere package required. Install with: pip install cohere"
                )

            api_key = self.config.get("api_key") if self.config else None
            self._client = cohere.Client(api_key) if api_key else cohere.Client()

        return self._client

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank using Cohere API."""
        if not results:
            return results

        client = self._get_client()
        config = self.get_config()

        # Get documents for reranking
        documents = [r.content for r in results]

        # Call Cohere rerank
        response = client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_n=config.top_k or len(results),
        )

        # Map results back by index
        reranked: list[SearchResult] = []
        for item in response.results:
            original = results[item.index]
            original.score = item.relevance_score
            reranked.append(original)

        if config.normalize_scores:
            reranked = self._normalize_scores(reranked)

        return reranked


@register_step(
    name="ScoreNormalizationStep",
    step_type=StepType.RERANKING,
    description="Normalize and adjust scores",
    version="1.0.0",
)
class ScoreNormalizationStep(BaseRerankingStep):
    """Simple score normalization and adjustment.

    Can apply min-max normalization, softmax, or custom scoring functions.
    """

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._method = config.get("method", "minmax") if config else "minmax"
        self._boost_recent = config.get("boost_recent", False) if config else False
        self._recency_weight = config.get("recency_weight", 0.1) if config else 0.1

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Normalize scores."""
        if not results:
            return results

        if self._method == "minmax":
            results = self._normalize_scores(results)
        elif self._method == "softmax":
            results = self._softmax_normalize(results)

        # Optional recency boost
        if self._boost_recent:
            results = self._apply_recency_boost(results)

        # Re-sort by final scores
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    def _softmax_normalize(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply softmax normalization."""
        import math

        scores = [r.score for r in results]
        max_score = max(scores)

        # Softmax with temperature
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)

        for result, exp_score in zip(results, exp_scores):
            result.score = exp_score / sum_exp

        return results

    def _apply_recency_boost(self, results: list[SearchResult]) -> list[SearchResult]:
        """Boost scores based on recency."""
        now = datetime.now(timezone.utc)

        for result in results:
            # Check for timestamp in metadata
            timestamp = result.metadata.get("created_at") or result.metadata.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)

                # Calculate age in days
                age_days = (now - timestamp).days if isinstance(timestamp, datetime) else 365

                # Apply decay (newer = higher boost)
                recency_factor = 1.0 / (1.0 + age_days / 30)  # Decay over ~30 days
                result.score = (1 - self._recency_weight) * result.score + self._recency_weight * recency_factor

        return results


@register_step(
    name="RecencyRerankingStep",
    step_type=StepType.RERANKING,
    description="Rerank based on document recency",
    version="1.0.0",
)
class RecencyRerankingStep(BaseRerankingStep):
    """Reranking that prioritizes recent documents.

    Combines relevance score with recency for time-sensitive queries.
    """

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        # Weight for recency vs relevance (1.0 = all recency, 0.0 = all relevance)
        self._recency_weight = config.get("recency_weight", 0.3) if config else 0.3
        self._decay_days = config.get("decay_days", 30) if config else 30

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank combining relevance and recency."""
        if not results:
            return results

        config = self.get_config()
        now = datetime.now(timezone.utc)

        for result in results:
            relevance_score = result.score

            # Get timestamp
            timestamp = (
                result.metadata.get("created_at") or
                result.metadata.get("timestamp") or
                result.metadata.get("updated_at")
            )

            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = None

            # Calculate recency score
            if timestamp and isinstance(timestamp, datetime):
                age_days = max(0, (now - timestamp).days)
                # Exponential decay
                recency_score = 2 ** (-age_days / self._decay_days)
            else:
                # No timestamp - use neutral score
                recency_score = 0.5

            # Combine scores
            result.score = (
                (1 - self._recency_weight) * relevance_score +
                self._recency_weight * recency_score
            )

        # Sort by combined score
        results.sort(key=lambda r: r.score, reverse=True)

        if config.normalize_scores:
            results = self._normalize_scores(results)

        return results


@register_step(
    name="MockRerankingStep",
    step_type=StepType.RERANKING,
    description="Mock reranker for testing",
    version="1.0.0",
)
class MockRerankingStep(BaseRerankingStep):
    """Mock reranking step for testing.

    Uses simple keyword matching for scoring.
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank using simple keyword matching."""
        if not results:
            return results

        config = self.get_config()
        query_terms = set(query.lower().split())

        for result in results:
            content_terms = set(result.content.lower().split())
            # Score based on term overlap
            overlap = len(query_terms & content_terms)
            result.score = overlap / max(len(query_terms), 1)

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        if config.normalize_scores:
            results = self._normalize_scores(results)

        return results
