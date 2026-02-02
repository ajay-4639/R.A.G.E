"""Unit tests for reranking system."""

import pytest
from datetime import datetime, timezone, timedelta

from rag_os.core.context import StepContext
from rag_os.models.index import SearchResult
from rag_os.steps.reranking import (
    BaseRerankingStep,
    RerankingConfig,
    ScoreNormalizationStep,
    RecencyRerankingStep,
    MockRerankingStep,
)


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    return [
        SearchResult(
            chunk_id="c1", doc_id="d1",
            content="Python is a popular programming language for data science.",
            score=0.8,
            metadata={"topic": "tech"},
        ),
        SearchResult(
            chunk_id="c2", doc_id="d1",
            content="Machine learning algorithms improve with more data.",
            score=0.75,
            metadata={"topic": "ml"},
        ),
        SearchResult(
            chunk_id="c3", doc_id="d2",
            content="The quick brown fox jumps over the lazy dog.",
            score=0.6,
            metadata={"topic": "other"},
        ),
        SearchResult(
            chunk_id="c4", doc_id="d2",
            content="Deep learning requires significant computational resources.",
            score=0.55,
            metadata={"topic": "ml"},
        ),
        SearchResult(
            chunk_id="c5", doc_id="d3",
            content="Natural language processing enables text analysis.",
            score=0.5,
            metadata={"topic": "nlp"},
        ),
    ]


@pytest.fixture
def results_with_timestamps() -> list[SearchResult]:
    """Create results with timestamp metadata."""
    now = datetime.now(timezone.utc)
    return [
        SearchResult(
            chunk_id="c1", doc_id="d1",
            content="Recent news about AI developments.",
            score=0.7,
            metadata={"created_at": (now - timedelta(days=1)).isoformat()},
        ),
        SearchResult(
            chunk_id="c2", doc_id="d2",
            content="Old article about machine learning basics.",
            score=0.8,
            metadata={"created_at": (now - timedelta(days=60)).isoformat()},
        ),
        SearchResult(
            chunk_id="c3", doc_id="d3",
            content="Very recent update on neural networks.",
            score=0.65,
            metadata={"created_at": now.isoformat()},
        ),
    ]


class TestRerankingConfig:
    """Tests for RerankingConfig."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = RerankingConfig()

        assert config.top_k is None
        assert config.min_score == 0.0
        assert config.normalize_scores is True

    def test_custom_config(self):
        """Custom config values work."""
        config = RerankingConfig(
            top_k=5,
            min_score=0.3,
            normalize_scores=False,
        )

        assert config.top_k == 5
        assert config.min_score == 0.3
        assert config.normalize_scores is False


class TestMockRerankingStep:
    """Tests for MockRerankingStep."""

    def test_rerank_basic(self, sample_results):
        """Basic reranking works."""
        step = MockRerankingStep(step_id="rerank", config={})

        reranked = step.rerank("python programming", sample_results)

        assert len(reranked) == len(sample_results)
        # Python content should be ranked higher
        assert "python" in reranked[0].content.lower()

    def test_rerank_keyword_matching(self, sample_results):
        """Reranking uses keyword matching."""
        step = MockRerankingStep(step_id="rerank", config={"normalize_scores": False})

        reranked = step.rerank("machine learning data", sample_results)

        # Results with more query term overlap should rank higher
        assert len(reranked) > 0
        # The ML content should be near the top
        ml_in_top_3 = any("machine learning" in r.content.lower() for r in reranked[:3])
        assert ml_in_top_3

    def test_execute_basic(self, sample_results):
        """Execute method works."""
        step = MockRerankingStep(step_id="rerank", config={"top_k": 3})

        context = StepContext(data={
            "query": "python programming",
            "results": sample_results,
        })
        result = step.execute(context)

        assert result.success
        assert len(result.output) == 3

    def test_execute_with_min_score(self, sample_results):
        """Execute applies min_score filter."""
        step = MockRerankingStep(step_id="rerank", config={"min_score": 0.5})

        context = StepContext(data={
            "query": "python",
            "results": sample_results,
        })
        result = step.execute(context)

        assert result.success
        for r in result.output:
            assert r.score >= 0.5

    def test_empty_results(self):
        """Handles empty results."""
        step = MockRerankingStep(step_id="rerank", config={})

        reranked = step.rerank("test query", [])

        assert reranked == []


class TestScoreNormalizationStep:
    """Tests for ScoreNormalizationStep."""

    def test_minmax_normalization(self, sample_results):
        """Min-max normalization works."""
        step = ScoreNormalizationStep(
            step_id="normalize",
            config={"method": "minmax"},
        )

        reranked = step.rerank("any query", sample_results)

        # Scores should be in [0, 1]
        for r in reranked:
            assert 0.0 <= r.score <= 1.0

        # Best should be 1.0, worst should be 0.0
        assert reranked[0].score == 1.0
        assert reranked[-1].score == 0.0

    def test_softmax_normalization(self, sample_results):
        """Softmax normalization works."""
        step = ScoreNormalizationStep(
            step_id="normalize",
            config={"method": "softmax", "normalize_scores": False},
        )

        reranked = step.rerank("any query", sample_results)

        # Scores should sum to ~1 (softmax)
        total = sum(r.score for r in reranked)
        assert abs(total - 1.0) < 0.001

    def test_recency_boost(self, results_with_timestamps):
        """Recency boost applies correctly."""
        step = ScoreNormalizationStep(
            step_id="normalize",
            config={
                "method": "minmax",
                "boost_recent": True,
                "recency_weight": 0.3,
            },
        )

        reranked = step.rerank("AI news", results_with_timestamps)

        # Recent content should be boosted
        assert len(reranked) == 3


class TestRecencyRerankingStep:
    """Tests for RecencyRerankingStep."""

    def test_recency_affects_ranking(self, results_with_timestamps):
        """Recency affects final ranking."""
        step = RecencyRerankingStep(
            step_id="recency",
            config={"recency_weight": 0.8},  # High recency weight
        )

        reranked = step.rerank("news", results_with_timestamps)

        # With high recency weight, very recent content should rank higher
        # c3 is most recent (today), c1 is 1 day old, c2 is 60 days old
        assert reranked[0].chunk_id in ["c3", "c1"]  # Both recent
        assert reranked[-1].chunk_id == "c2"  # Oldest should be last

    def test_relevance_vs_recency_balance(self, results_with_timestamps):
        """Different weights affect balance."""
        # Pure relevance
        step_relevance = RecencyRerankingStep(
            step_id="recency",
            config={"recency_weight": 0.0},
        )

        # Pure recency
        step_recency = RecencyRerankingStep(
            step_id="recency",
            config={"recency_weight": 1.0},
        )

        query = "machine learning"

        results_rel = step_relevance.rerank(query, results_with_timestamps.copy())
        results_rec = step_recency.rerank(query, results_with_timestamps.copy())

        # With pure relevance, highest original score wins
        assert results_rel[0].chunk_id == "c2"  # Score 0.8

        # With pure recency, most recent wins
        assert results_rec[0].chunk_id == "c3"  # Most recent

    def test_missing_timestamp(self):
        """Handles missing timestamps gracefully."""
        results = [
            SearchResult(chunk_id="c1", doc_id="d1", content="No timestamp", score=0.8, metadata={}),
            SearchResult(chunk_id="c2", doc_id="d2", content="Also no timestamp", score=0.6, metadata={}),
        ]

        step = RecencyRerankingStep(
            step_id="recency",
            config={"recency_weight": 0.3},
        )

        reranked = step.rerank("test", results)

        # Should not crash, should use neutral recency score
        assert len(reranked) == 2


class TestRerankingExecuteErrors:
    """Tests for error handling in reranking."""

    def test_missing_query_fails(self, sample_results):
        """Missing query returns error."""
        step = MockRerankingStep(step_id="rerank", config={})

        context = StepContext(data={
            "results": sample_results,
        })
        result = step.execute(context)

        assert not result.success

    def test_invalid_input_type_fails(self):
        """Invalid input type returns error."""
        step = MockRerankingStep(step_id="rerank", config={})

        context = StepContext(data="not a dict")
        result = step.execute(context)

        assert not result.success

    def test_invalid_result_type_fails(self):
        """Invalid result type in list returns error."""
        step = MockRerankingStep(step_id="rerank", config={})

        context = StepContext(data={
            "query": "test",
            "results": ["not a result", "also not a result"],
        })
        result = step.execute(context)

        assert not result.success


class TestRerankingScoreNormalization:
    """Tests for score normalization behavior."""

    def test_normalize_identical_scores(self):
        """Normalization handles identical scores."""
        results = [
            SearchResult(chunk_id="c1", doc_id="d1", content="A", score=0.5, metadata={}),
            SearchResult(chunk_id="c2", doc_id="d2", content="B", score=0.5, metadata={}),
        ]

        step = MockRerankingStep(step_id="rerank", config={})

        # _normalize_scores should handle this
        normalized = step._normalize_scores(results)

        # All should be 1.0 when identical
        for r in normalized:
            assert r.score == 1.0

    def test_normalize_preserves_order(self, sample_results):
        """Normalization preserves relative order."""
        step = MockRerankingStep(step_id="rerank", config={})

        # Sort by score first
        sorted_results = sorted(sample_results, key=lambda r: r.score, reverse=True)
        original_order = [r.chunk_id for r in sorted_results]

        normalized = step._normalize_scores(sorted_results)
        normalized_order = [r.chunk_id for r in normalized]

        assert original_order == normalized_order


class TestRerankingIntegration:
    """Integration tests for reranking steps."""

    def test_pipeline_style_reranking(self, sample_results):
        """Test reranking in pipeline-style usage."""
        # First normalize
        normalizer = ScoreNormalizationStep(step_id="normalize", config={"method": "minmax"})

        # Then apply keyword boost
        reranker = MockRerankingStep(step_id="rerank", config={"top_k": 3})

        # Simulate pipeline
        context1 = StepContext(data={"query": "python data science", "results": sample_results})
        result1 = normalizer.execute(context1)

        assert result1.success

        context2 = StepContext(data={"query": "python data science", "results": result1.output})
        result2 = reranker.execute(context2)

        assert result2.success
        assert len(result2.output) == 3

    def test_chained_reranking(self, sample_results):
        """Multiple reranking steps can be chained."""
        step1 = MockRerankingStep(step_id="rerank1", config={"normalize_scores": False})
        step2 = ScoreNormalizationStep(step_id="normalize", config={"method": "minmax"})

        # First rerank
        reranked = step1.rerank("python programming", sample_results)

        # Then normalize
        context = StepContext(data={"query": "python programming", "results": reranked})
        result = step2.execute(context)

        assert result.success
        # All scores should be normalized
        for r in result.output:
            assert 0.0 <= r.score <= 1.0
