"""Retrieval step implementations for RAG OS."""

from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.index import SearchResult
from rag_os.steps.retrieval.base import BaseRetrievalStep
from rag_os.steps.indexing.base import BaseIndex
from rag_os.steps.embedding.base import BaseEmbeddingStep


@register_step(
    name="VectorRetrievalStep",
    step_type=StepType.RETRIEVAL,
    description="Vector similarity-based retrieval",
    version="1.0.0",
)
class VectorRetrievalStep(BaseRetrievalStep):
    """Retrieval step using vector similarity search.

    Embeds the query and searches the index for similar chunks.
    """

    def __init__(
        self,
        step_id: str,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
        embedder: BaseEmbeddingStep | None = None,
    ):
        super().__init__(step_id, config=config, index=index, embedder=embedder)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve chunks using vector similarity.

        Args:
            query: Query text to search for
            top_k: Number of results to return
            filters: Metadata filters

        Returns:
            List of search results
        """
        config = self.get_config()
        k = top_k or config.top_k

        # Merge filters
        merged_filters = {**(config.filters or {}), **(filters or {})}

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Search the index
        results = self.index.search(
            query_embedding=query_embedding,
            top_k=k,
            filters=merged_filters if merged_filters else None,
        )

        return results


@register_step(
    name="HybridRetrievalStep",
    step_type=StepType.RETRIEVAL,
    description="Hybrid retrieval combining dense and sparse",
    version="1.0.0",
)
class HybridRetrievalStep(BaseRetrievalStep):
    """Hybrid retrieval combining vector search with keyword matching.

    Combines dense embeddings with sparse representations (e.g., BM25)
    for improved retrieval quality.
    """

    def __init__(
        self,
        step_id: str,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
        embedder: BaseEmbeddingStep | None = None,
    ):
        super().__init__(step_id, config=config, index=index, embedder=embedder)
        # Weight for dense vs sparse (1.0 = all dense, 0.0 = all sparse)
        self._alpha = config.get("alpha", 0.7) if config else 0.7

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using hybrid search.

        Combines vector similarity with keyword matching.
        """
        config = self.get_config()
        k = top_k or config.top_k

        merged_filters = {**(config.filters or {}), **(filters or {})}

        # Get vector results
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.index.search(
            query_embedding=query_embedding,
            top_k=k * 2,  # Get more for merging
            filters=merged_filters if merged_filters else None,
        )

        # Simple keyword scoring (basic implementation)
        # In production, this would use BM25 or similar
        query_terms = set(query.lower().split())
        keyword_scores: dict[str, float] = {}

        for result in vector_results:
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                keyword_scores[result.chunk_id] = overlap / len(query_terms)

        # Combine scores
        combined: dict[str, tuple[float, SearchResult]] = {}
        for result in vector_results:
            vector_score = result.score
            keyword_score = keyword_scores.get(result.chunk_id, 0.0)

            # Weighted combination
            combined_score = (
                self._alpha * vector_score +
                (1 - self._alpha) * keyword_score
            )

            combined[result.chunk_id] = (combined_score, result)

        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x[0],
            reverse=True
        )[:k]

        # Update scores in results
        final_results = []
        for score, result in sorted_results:
            result.score = score
            final_results.append(result)

        return final_results


@register_step(
    name="MultiQueryRetrievalStep",
    step_type=StepType.RETRIEVAL,
    description="Multi-query retrieval for improved recall",
    version="1.0.0",
)
class MultiQueryRetrievalStep(BaseRetrievalStep):
    """Multi-query retrieval that generates query variations.

    Generates multiple query variations and combines results
    for improved recall.
    """

    def __init__(
        self,
        step_id: str,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
        embedder: BaseEmbeddingStep | None = None,
    ):
        super().__init__(step_id, config=config, index=index, embedder=embedder)
        self._query_generator = config.get("query_generator") if config else None

    def generate_queries(self, query: str) -> list[str]:
        """Generate query variations.

        Override this or provide a query_generator in config
        for more sophisticated query expansion.

        Args:
            query: Original query

        Returns:
            List of query variations including original
        """
        if self._query_generator:
            return self._query_generator(query)

        # Simple variations by default
        queries = [query]

        # Add question form if not already a question
        if not query.strip().endswith("?"):
            queries.append(f"What is {query}?")

        # Add "about" form
        queries.append(f"Information about {query}")

        return queries

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using multiple query variations.

        Searches with each query variation and combines results
        using reciprocal rank fusion.
        """
        config = self.get_config()
        k = top_k or config.top_k

        merged_filters = {**(config.filters or {}), **(filters or {})}

        # Generate query variations
        queries = self.generate_queries(query)

        # Collect results for each query
        all_results: dict[str, list[tuple[int, SearchResult]]] = {}

        for query_var in queries:
            query_embedding = self.embedder.embed_query(query_var)
            results = self.index.search(
                query_embedding=query_embedding,
                top_k=k,
                filters=merged_filters if merged_filters else None,
            )

            for rank, result in enumerate(results):
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = []
                all_results[result.chunk_id].append((rank, result))

        # Reciprocal Rank Fusion
        k_constant = 60  # RRF constant
        fused_scores: dict[str, tuple[float, SearchResult]] = {}

        for chunk_id, rankings in all_results.items():
            # RRF score
            rrf_score = sum(1 / (k_constant + rank) for rank, _ in rankings)
            # Use the first result's data
            result = rankings[0][1]
            fused_scores[chunk_id] = (rrf_score, result)

        # Sort by fused score
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )[:k]

        # Update scores
        final_results = []
        for score, result in sorted_results:
            result.score = score
            final_results.append(result)

        return final_results


@register_step(
    name="MMRRetrievalStep",
    step_type=StepType.RETRIEVAL,
    description="Maximum Marginal Relevance retrieval for diversity",
    version="1.0.0",
)
class MMRRetrievalStep(BaseRetrievalStep):
    """Maximum Marginal Relevance retrieval for diverse results.

    Balances relevance with diversity to avoid redundant results.
    """

    def __init__(
        self,
        step_id: str,
        config: dict[str, Any] | None = None,
        index: BaseIndex | None = None,
        embedder: BaseEmbeddingStep | None = None,
    ):
        super().__init__(step_id, config=config, index=index, embedder=embedder)
        # Lambda for relevance vs diversity (1.0 = all relevance)
        self._lambda = config.get("lambda", 0.7) if config else 0.7

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using MMR for diverse results."""
        import math

        config = self.get_config()
        k = top_k or config.top_k

        merged_filters = {**(config.filters or {}), **(filters or {})}

        # Get initial candidates
        query_embedding = self.embedder.embed_query(query)
        candidates = self.index.search(
            query_embedding=query_embedding,
            top_k=k * 3,  # Get more candidates for MMR
            filters=merged_filters if merged_filters else None,
        )

        if not candidates:
            return []

        # MMR selection
        selected: list[SearchResult] = []
        candidate_embeddings: dict[str, list[float]] = {}

        # Get embeddings for candidates (from index if available)
        for c in candidates:
            if c.embedding:
                candidate_embeddings[c.chunk_id] = c.embedding

        # Greedy MMR selection
        remaining = list(candidates)

        while len(selected) < k and remaining:
            best_score = float("-inf")
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.score

                # Diversity score (max similarity to already selected)
                diversity = 0.0
                if selected and candidate.chunk_id in candidate_embeddings:
                    for sel in selected:
                        if sel.chunk_id in candidate_embeddings:
                            sim = self._cosine_similarity(
                                candidate_embeddings[candidate.chunk_id],
                                candidate_embeddings[sel.chunk_id],
                            )
                            diversity = max(diversity, sim)

                # MMR score
                mmr_score = self._lambda * relevance - (1 - self._lambda) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Add best candidate
            best_candidate = remaining.pop(best_idx)
            best_candidate.score = best_score
            selected.append(best_candidate)

        return selected

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        import math
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
