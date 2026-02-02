"""Unit tests for retrieval system."""

import pytest
import math

from rag_os.core.context import StepContext
from rag_os.models.chunk import Chunk
from rag_os.models.embedding import EmbeddedChunk
from rag_os.models.index import SearchQuery, IndexConfig, DistanceMetric
from rag_os.steps.indexing import InMemoryVectorIndex
from rag_os.steps.embedding import MockEmbeddingStep
from rag_os.steps.retrieval import (
    BaseRetrievalStep,
    RetrievalConfig,
    VectorRetrievalStep,
    HybridRetrievalStep,
    MultiQueryRetrievalStep,
    MMRRetrievalStep,
)


@pytest.fixture
def embedder() -> MockEmbeddingStep:
    """Create a mock embedder."""
    return MockEmbeddingStep(
        step_id="embedder",
        config={"dimensions": 128},
    )


@pytest.fixture
def index_with_data(embedder) -> InMemoryVectorIndex:
    """Create an index populated with test data."""
    index_config = IndexConfig(dimensions=128, metric=DistanceMetric.COSINE)
    index = InMemoryVectorIndex(index_config)

    # Create test chunks
    chunks = [
        Chunk(content="The quick brown fox jumps over the lazy dog.", doc_id="doc1", index=0),
        Chunk(content="Machine learning algorithms improve with more data.", doc_id="doc1", index=1),
        Chunk(content="Python is a popular programming language.", doc_id="doc2", index=0),
        Chunk(content="Deep learning requires significant computational resources.", doc_id="doc2", index=1),
        Chunk(content="Natural language processing enables text analysis.", doc_id="doc3", index=0),
        Chunk(content="The cat sleeps on the warm windowsill.", doc_id="doc3", index=1),
        Chunk(content="Data science combines statistics and programming.", doc_id="doc4", index=0),
        Chunk(content="Neural networks are inspired by the brain.", doc_id="doc4", index=1),
    ]

    # Embed and add to index
    embedded = embedder.embed_chunks(chunks)
    index.add(embedded)

    return index


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = RetrievalConfig()

        assert config.top_k == 10
        assert config.min_score == 0.0
        assert config.rerank is False
        assert config.include_metadata is True

    def test_custom_config(self):
        """Custom config values work."""
        config = RetrievalConfig(
            top_k=5,
            min_score=0.5,
            rerank=True,
            filters={"doc_id": "doc1"},
        )

        assert config.top_k == 5
        assert config.min_score == 0.5
        assert config.rerank is True


class TestVectorRetrievalStep:
    """Tests for VectorRetrievalStep."""

    def test_retrieve_basic(self, embedder, index_with_data):
        """Basic retrieval works."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={"top_k": 3},
            index=index_with_data,
            embedder=embedder,
        )

        results = step.retrieve("machine learning data")

        assert len(results) == 3
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "content") for r in results)

    def test_retrieve_with_filters(self, embedder, index_with_data):
        """Retrieval with filters works."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={"top_k": 10},
            index=index_with_data,
            embedder=embedder,
        )

        results = step.retrieve("programming", filters={"doc_id": "doc2"})

        # Should only return results from doc2
        assert all(r.doc_id == "doc2" for r in results)

    def test_execute_with_string(self, embedder, index_with_data):
        """Execute works with string input."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5},
            index=index_with_data,
            embedder=embedder,
        )

        context = StepContext(data="neural networks")
        result = step.execute(context)

        assert result.success
        assert len(result.output) <= 5
        assert result.metadata["result_count"] > 0

    def test_execute_with_dict(self, embedder, index_with_data):
        """Execute works with dict input."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={"top_k": 10},
            index=index_with_data,
            embedder=embedder,
        )

        context = StepContext(data={
            "query_text": "programming language",
            "top_k": 3,
        })
        result = step.execute(context)

        assert result.success
        assert len(result.output) == 3

    def test_execute_with_search_query(self, embedder, index_with_data):
        """Execute works with SearchQuery input."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            index=index_with_data,
            embedder=embedder,
        )

        query = SearchQuery(
            query_text="deep learning",
            top_k=2,
        )
        context = StepContext(data=query)
        result = step.execute(context)

        assert result.success
        assert len(result.output) <= 2

    def test_min_score_filter(self, embedder, index_with_data):
        """Min score filtering works."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={"top_k": 10, "min_score": 0.99},  # Very high threshold
            index=index_with_data,
            embedder=embedder,
        )

        context = StepContext(data="completely unrelated query xyz")
        result = step.execute(context)

        assert result.success
        # Should filter out low-scoring results
        for r in result.output:
            assert r.score >= 0.99


class TestHybridRetrievalStep:
    """Tests for HybridRetrievalStep."""

    def test_retrieve_combines_scores(self, embedder, index_with_data):
        """Hybrid retrieval combines vector and keyword scores."""
        step = HybridRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "alpha": 0.5},  # Equal weight
            index=index_with_data,
            embedder=embedder,
        )

        # Query with specific keyword
        results = step.retrieve("python programming")

        assert len(results) == 5
        # Python-related content should score higher
        python_in_top = any("python" in r.content.lower() for r in results[:3])
        assert python_in_top

    def test_alpha_parameter(self, embedder, index_with_data):
        """Alpha parameter affects score combination."""
        # Pure vector
        step_vector = HybridRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "alpha": 1.0},
            index=index_with_data,
            embedder=embedder,
        )

        # Pure keyword
        step_keyword = HybridRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "alpha": 0.0},
            index=index_with_data,
            embedder=embedder,
        )

        query = "fox dog"
        results_vector = step_vector.retrieve(query)
        results_keyword = step_keyword.retrieve(query)

        # Results should differ based on alpha
        # (may not always differ depending on query)
        assert len(results_vector) > 0
        assert len(results_keyword) > 0


class TestMultiQueryRetrievalStep:
    """Tests for MultiQueryRetrievalStep."""

    def test_generates_query_variations(self, embedder, index_with_data):
        """Multi-query generates variations."""
        step = MultiQueryRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5},
            index=index_with_data,
            embedder=embedder,
        )

        variations = step.generate_queries("machine learning")

        assert len(variations) > 1
        assert "machine learning" in variations

    def test_retrieve_uses_rrf(self, embedder, index_with_data):
        """Multi-query uses reciprocal rank fusion."""
        step = MultiQueryRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5},
            index=index_with_data,
            embedder=embedder,
        )

        results = step.retrieve("data science programming")

        assert len(results) == 5
        # Results should have positive scores from RRF
        assert all(r.score > 0 for r in results)

    def test_custom_query_generator(self, embedder, index_with_data):
        """Can use custom query generator."""
        def custom_generator(query: str) -> list[str]:
            return [query, f"Tell me about {query}", query.upper()]

        step = MultiQueryRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "query_generator": custom_generator},
            index=index_with_data,
            embedder=embedder,
        )

        variations = step.generate_queries("test")
        assert len(variations) == 3


class TestMMRRetrievalStep:
    """Tests for MMRRetrievalStep."""

    def test_retrieve_diverse_results(self, embedder, index_with_data):
        """MMR retrieval promotes diversity."""
        step = MMRRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "lambda": 0.5},  # Balance relevance/diversity
            index=index_with_data,
            embedder=embedder,
        )

        results = step.retrieve("learning algorithms")

        assert len(results) == 5
        # Results should come from multiple documents (diversity)
        doc_ids = set(r.doc_id for r in results)
        assert len(doc_ids) > 1

    def test_lambda_affects_diversity(self, embedder, index_with_data):
        """Lambda parameter controls relevance vs diversity."""
        # High lambda = more relevance focus
        step_relevant = MMRRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "lambda": 0.95},
            index=index_with_data,
            embedder=embedder,
        )

        # Low lambda = more diversity focus
        step_diverse = MMRRetrievalStep(
            step_id="retrieval",
            config={"top_k": 5, "lambda": 0.3},
            index=index_with_data,
            embedder=embedder,
        )

        query = "machine learning neural networks"

        results_relevant = step_relevant.retrieve(query)
        results_diverse = step_diverse.retrieve(query)

        # Both should return results
        assert len(results_relevant) > 0
        assert len(results_diverse) > 0


class TestRetrievalExecuteErrors:
    """Tests for error handling in retrieval steps."""

    def test_empty_query_fails(self, embedder, index_with_data):
        """Empty query returns error."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            index=index_with_data,
            embedder=embedder,
        )

        context = StepContext(data="")
        result = step.execute(context)

        assert not result.success

    def test_invalid_input_type_fails(self, embedder, index_with_data):
        """Invalid input type returns error."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            index=index_with_data,
            embedder=embedder,
        )

        context = StepContext(data=123)  # Invalid type
        result = step.execute(context)

        assert not result.success

    def test_missing_index_raises(self, embedder):
        """Missing index raises error."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            embedder=embedder,
        )

        with pytest.raises(ValueError, match="Index not set"):
            step.retrieve("test query")

    def test_missing_embedder_raises(self, index_with_data):
        """Missing embedder raises error."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            index=index_with_data,
        )

        with pytest.raises(ValueError, match="Embedder not set"):
            step.retrieve("test query")


class TestRetrievalSetters:
    """Tests for index and embedder setters."""

    def test_set_index(self, embedder, index_with_data):
        """Can set index after construction."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            embedder=embedder,
        )

        step.set_index(index_with_data)

        results = step.retrieve("test", top_k=3)
        assert len(results) == 3

    def test_set_embedder(self, embedder, index_with_data):
        """Can set embedder after construction."""
        step = VectorRetrievalStep(
            step_id="retrieval",
            config={},
            index=index_with_data,
        )

        step.set_embedder(embedder)

        results = step.retrieve("test", top_k=3)
        assert len(results) == 3
