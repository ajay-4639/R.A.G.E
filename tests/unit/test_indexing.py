"""Unit tests for indexing system."""

import pytest
import math

from rag_os.core.context import StepContext
from rag_os.models.chunk import Chunk
from rag_os.models.embedding import EmbeddedChunk
from rag_os.models.index import (
    IndexType,
    DistanceMetric,
    SearchResult,
    SearchQuery,
    SearchResponse,
    IndexStats,
    IndexConfig,
)
from rag_os.steps.indexing import (
    BaseIndex,
    InMemoryVectorIndex,
    IndexingStep,
)


@pytest.fixture
def sample_embedded_chunks() -> list[EmbeddedChunk]:
    """Create sample embedded chunks for testing."""
    chunks = [
        Chunk(content="The cat sat on the mat.", doc_id="doc1", index=0, metadata={"topic": "animals"}),
        Chunk(content="Dogs are loyal companions.", doc_id="doc1", index=1, metadata={"topic": "animals"}),
        Chunk(content="Python is a programming language.", doc_id="doc2", index=0, metadata={"topic": "tech"}),
        Chunk(content="Machine learning is fascinating.", doc_id="doc2", index=1, metadata={"topic": "tech"}),
        Chunk(content="Birds can fly high.", doc_id="doc3", index=0, metadata={"topic": "animals"}),
    ]

    # Create simple embeddings (unit vectors for testing)
    embedded = []
    for i, chunk in enumerate(chunks):
        # Create a simple embedding - different angle for each
        angle = i * 0.3
        embedding = [math.cos(angle + j * 0.1) for j in range(128)]
        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / norm for x in embedding]

        embedded.append(EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            model_name="test-model",
            dimensions=128,
        ))

    return embedded


@pytest.fixture
def index_config() -> IndexConfig:
    """Create index configuration."""
    return IndexConfig(
        name="test-index",
        dimensions=128,
        metric=DistanceMetric.COSINE,
    )


class TestIndexModels:
    """Tests for index models."""

    def test_search_result_creation(self):
        """Can create a search result."""
        result = SearchResult(
            chunk_id="c1",
            doc_id="d1",
            content="Test content",
            score=0.95,
            metadata={"key": "value"},
        )

        assert result.chunk_id == "c1"
        assert result.score == 0.95

    def test_search_result_serialization(self):
        """SearchResult can be serialized."""
        result = SearchResult(
            chunk_id="c1",
            doc_id="d1",
            content="Test",
            score=0.9,
        )

        data = result.to_dict()
        restored = SearchResult.from_dict(data)

        assert restored.chunk_id == result.chunk_id
        assert restored.score == result.score

    def test_search_query_defaults(self):
        """SearchQuery has sensible defaults."""
        query = SearchQuery(query_text="test query")

        assert query.top_k == 10
        assert query.min_score == 0.0
        assert query.include_metadata is True

    def test_search_response_properties(self):
        """SearchResponse properties work."""
        results = [
            SearchResult(chunk_id=f"c{i}", doc_id="d1", content=f"Content {i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]
        query = SearchQuery(query_text="test")

        response = SearchResponse(
            results=results,
            query=query,
            total_count=5,
        )

        assert response.count == 5
        assert len(response.top(2)) == 2
        assert response.top(1)[0].chunk_id == "c0"

    def test_index_stats(self):
        """IndexStats can be created and serialized."""
        stats = IndexStats(
            total_chunks=100,
            total_documents=10,
            dimensions=384,
        )

        assert stats.total_chunks == 100

        data = stats.to_dict()
        assert data["total_chunks"] == 100
        assert data["index_type"] == "dense"

    def test_index_config(self):
        """IndexConfig can be created and serialized."""
        config = IndexConfig(
            name="my-index",
            dimensions=512,
            metric=DistanceMetric.DOT_PRODUCT,
        )

        data = config.to_dict()
        restored = IndexConfig.from_dict(data)

        assert restored.name == "my-index"
        assert restored.dimensions == 512
        assert restored.metric == DistanceMetric.DOT_PRODUCT


class TestInMemoryVectorIndex:
    """Tests for InMemoryVectorIndex."""

    def test_add_chunks(self, sample_embedded_chunks, index_config):
        """Can add chunks to the index."""
        index = InMemoryVectorIndex(index_config)

        count = index.add(sample_embedded_chunks)

        assert count == 5
        stats = index.get_stats()
        assert stats.total_chunks == 5
        assert stats.total_documents == 3

    def test_search_basic(self, sample_embedded_chunks, index_config):
        """Basic search works."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        # Search with the first chunk's embedding
        query_embedding = sample_embedded_chunks[0].embedding
        results = index.search(query_embedding, top_k=3)

        assert len(results) == 3
        # Most similar should be itself
        assert results[0].chunk_id == sample_embedded_chunks[0].chunk_id
        assert results[0].score > 0.99  # Should be ~1.0 for identical

    def test_search_with_filters(self, sample_embedded_chunks, index_config):
        """Search with metadata filters works."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        query_embedding = sample_embedded_chunks[0].embedding
        results = index.search(
            query_embedding,
            top_k=10,
            filters={"topic": "tech"},
        )

        # Should only return tech-related chunks
        assert len(results) == 2
        assert all(r.metadata.get("topic") == "tech" for r in results)

    def test_search_filter_by_doc_id(self, sample_embedded_chunks, index_config):
        """Can filter by doc_id."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        query_embedding = sample_embedded_chunks[0].embedding
        results = index.search(
            query_embedding,
            top_k=10,
            filters={"doc_id": "doc1"},
        )

        assert len(results) == 2
        assert all(r.doc_id == "doc1" for r in results)

    def test_delete_chunks(self, sample_embedded_chunks, index_config):
        """Can delete chunks from the index."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        chunk_ids = [sample_embedded_chunks[0].chunk_id, sample_embedded_chunks[1].chunk_id]
        deleted = index.delete(chunk_ids)

        assert deleted == 2
        stats = index.get_stats()
        assert stats.total_chunks == 3

    def test_clear_index(self, sample_embedded_chunks, index_config):
        """Can clear all data from index."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)
        index.clear()

        stats = index.get_stats()
        assert stats.total_chunks == 0
        assert stats.total_documents == 0

    def test_get_chunk(self, sample_embedded_chunks, index_config):
        """Can retrieve a specific chunk."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        chunk_id = sample_embedded_chunks[2].chunk_id
        chunk = index.get_chunk(chunk_id)

        assert chunk is not None
        assert chunk.content == "Python is a programming language."

    def test_upsert_behavior(self, sample_embedded_chunks, index_config):
        """Adding same chunk twice updates it."""
        index = InMemoryVectorIndex(index_config)
        index.add([sample_embedded_chunks[0]])
        index.add([sample_embedded_chunks[0]])  # Add same chunk again

        stats = index.get_stats()
        assert stats.total_chunks == 1  # Should not duplicate

    def test_search_with_query_object(self, sample_embedded_chunks, index_config):
        """Search with SearchQuery object works."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        query = SearchQuery(
            query_text="cats",
            query_embedding=sample_embedded_chunks[0].embedding,
            top_k=2,
            min_score=0.5,
        )

        response = index.search_with_query(query)

        assert isinstance(response, SearchResponse)
        assert response.count <= 2
        assert all(r.score >= 0.5 for r in response.results)

    def test_different_distance_metrics(self, sample_embedded_chunks):
        """Different distance metrics work."""
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            config = IndexConfig(dimensions=128, metric=metric)
            index = InMemoryVectorIndex(config)
            index.add(sample_embedded_chunks)

            results = index.search(sample_embedded_chunks[0].embedding, top_k=3)

            assert len(results) == 3
            # Self should always be most similar
            assert results[0].chunk_id == sample_embedded_chunks[0].chunk_id


class TestIndexingStep:
    """Tests for IndexingStep."""

    def test_execute_basic(self, sample_embedded_chunks):
        """Execute adds chunks to index."""
        step = IndexingStep(
            step_id="indexing",
            config={"dimensions": 128},
        )

        context = StepContext(data=sample_embedded_chunks)
        result = step.execute(context)

        assert result.success
        assert result.metadata["chunks_indexed"] == 5

    def test_execute_with_custom_index(self, sample_embedded_chunks, index_config):
        """Can use custom index."""
        index = InMemoryVectorIndex(index_config)
        step = IndexingStep(
            step_id="indexing",
            config={},
            index=index,
        )

        context = StepContext(data=sample_embedded_chunks)
        result = step.execute(context)

        assert result.success
        assert index.get_stats().total_chunks == 5

    def test_search_after_indexing(self, sample_embedded_chunks):
        """Can search after indexing."""
        step = IndexingStep(
            step_id="indexing",
            config={"dimensions": 128},
        )

        context = StepContext(data=sample_embedded_chunks)
        step.execute(context)

        # Search
        results = step.search(sample_embedded_chunks[0].embedding, top_k=3)

        assert len(results) == 3
        assert results[0].score > 0.9

    def test_execute_empty_input(self):
        """Handles empty input gracefully."""
        step = IndexingStep(step_id="indexing", config={})

        context = StepContext(data=[])
        result = step.execute(context)

        assert result.success
        assert result.metadata["chunks_indexed"] == 0

    def test_upsert_method(self, sample_embedded_chunks):
        """Upsert method works."""
        step = IndexingStep(
            step_id="indexing",
            config={"dimensions": 128},
        )

        count = step.upsert(sample_embedded_chunks[:2])
        assert count == 2

        count = step.upsert(sample_embedded_chunks[2:])
        assert count == 3

        stats = step.index.get_stats()
        assert stats.total_chunks == 5


class TestIndexFiltering:
    """Tests for index filtering capabilities."""

    def test_filter_by_multiple_doc_ids(self, sample_embedded_chunks, index_config):
        """Can filter by multiple doc IDs."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        results = index.search(
            sample_embedded_chunks[0].embedding,
            top_k=10,
            filters={"doc_ids": ["doc1", "doc3"]},
        )

        assert len(results) == 3  # 2 from doc1 + 1 from doc3
        assert all(r.doc_id in ["doc1", "doc3"] for r in results)

    def test_filter_by_list_value(self, sample_embedded_chunks, index_config):
        """Can filter by list of allowed values."""
        index = InMemoryVectorIndex(index_config)
        index.add(sample_embedded_chunks)

        results = index.search(
            sample_embedded_chunks[0].embedding,
            top_k=10,
            filters={"topic": ["tech"]},  # List with single value
        )

        assert len(results) == 2
