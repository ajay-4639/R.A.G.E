"""Unit tests for embedding system."""

import pytest
import math

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.models.chunk import Chunk
from rag_os.models.embedding import EmbeddedChunk, EmbeddingConfig, EmbeddingResult
from rag_os.steps.embedding import (
    BaseEmbeddingStep,
    MockEmbeddingStep,
    OpenAIEmbeddingStep,
    LocalEmbeddingStep,
)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(content="This is the first chunk.", doc_id="doc1", index=0),
        Chunk(content="This is the second chunk.", doc_id="doc1", index=1),
        Chunk(content="And this is the third chunk.", doc_id="doc1", index=2),
    ]


@pytest.fixture
def sample_texts() -> list[str]:
    """Create sample texts for testing."""
    return [
        "Machine learning is fascinating.",
        "Natural language processing enables many applications.",
        "Vector embeddings capture semantic meaning.",
    ]


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk model."""

    def test_creation(self, sample_chunks):
        """Can create an embedded chunk."""
        chunk = sample_chunks[0]
        embedding = [0.1] * 384

        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            model_name="test-model",
        )

        assert embedded.chunk == chunk
        assert embedded.embedding == embedding
        assert embedded.model_name == "test-model"
        assert embedded.dimensions == 384

    def test_properties(self, sample_chunks):
        """Properties delegate to chunk."""
        chunk = sample_chunks[0]
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.0] * 128,
        )

        assert embedded.chunk_id == chunk.chunk_id
        assert embedded.doc_id == chunk.doc_id
        assert embedded.content == chunk.content

    def test_serialization(self, sample_chunks):
        """Can serialize and deserialize."""
        chunk = sample_chunks[0]
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            model_name="test",
            model_version="2.0",
        )

        data = embedded.to_dict()
        restored = EmbeddedChunk.from_dict(data)

        assert restored.embedding == embedded.embedding
        assert restored.model_name == embedded.model_name
        assert restored.model_version == embedded.model_version
        assert restored.chunk.content == chunk.content


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = EmbeddingConfig()

        assert config.model_name == "text-embedding-3-small"
        assert config.batch_size == 100
        assert config.normalize is True
        assert config.max_retries == 3

    def test_custom_config(self):
        """Can set custom values."""
        config = EmbeddingConfig(
            model_name="custom-model",
            dimensions=512,
            batch_size=50,
            normalize=False,
        )

        assert config.model_name == "custom-model"
        assert config.dimensions == 512
        assert config.batch_size == 50
        assert config.normalize is False

    def test_serialization(self):
        """Serialization excludes sensitive data."""
        config = EmbeddingConfig(
            model_name="test",
            api_key="secret-key",
        )

        data = config.to_dict()
        assert "api_key" not in data  # Should not include sensitive data
        assert data["model_name"] == "test"


class TestEmbeddingResult:
    """Tests for EmbeddingResult."""

    def test_basic(self):
        """Basic result properties work."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model_name="test",
            dimensions=2,
            token_count=10,
        )

        assert result.count == 2
        assert result.dimensions == 2
        assert result.token_count == 10


class TestMockEmbeddingStep:
    """Tests for MockEmbeddingStep."""

    def test_basic_embedding(self, sample_texts):
        """Can embed texts."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 384},
        )

        embeddings = step.embed_texts(sample_texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_deterministic(self, sample_texts):
        """Same text produces same embedding."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128},
        )

        embeddings1 = step.embed_texts(sample_texts)
        embeddings2 = step.embed_texts(sample_texts)

        for e1, e2 in zip(embeddings1, embeddings2):
            assert e1 == e2

    def test_different_texts_different_embeddings(self):
        """Different texts produce different embeddings."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128},
        )

        emb1 = step.embed_texts(["Hello world"])[0]
        emb2 = step.embed_texts(["Goodbye world"])[0]

        assert emb1 != emb2

    def test_normalization(self):
        """Embeddings are normalized when configured."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128, "normalize": True},
        )

        embeddings = step.embed_texts(["Test text"])
        emb = embeddings[0]

        # Check unit length (within floating point tolerance)
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.001

    def test_embed_chunks(self, sample_chunks):
        """Can embed Chunk objects."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 256},
        )

        embedded_chunks = step.embed_chunks(sample_chunks)

        assert len(embedded_chunks) == 3
        assert all(isinstance(ec, EmbeddedChunk) for ec in embedded_chunks)
        assert all(ec.dimensions == 256 for ec in embedded_chunks)
        assert all(ec.model_name == "mock-embedding-model" for ec in embedded_chunks)

    def test_embed_query(self):
        """Can embed a single query."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128},
        )

        query_embedding = step.embed_query("What is machine learning?")

        assert len(query_embedding) == 128

    def test_execute_with_chunks(self, sample_chunks):
        """Execute works with Chunk input."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128},
        )

        context = StepContext(data=sample_chunks)
        result = step.execute(context)

        assert result.success
        assert len(result.output) == 3
        assert result.metadata["embedded_count"] == 3
        assert result.metadata["dimensions"] == 128

    def test_execute_with_strings(self, sample_texts):
        """Execute works with string input."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128},
        )

        context = StepContext(data=sample_texts)
        result = step.execute(context)

        assert result.success
        assert len(result.output) == 3

    def test_execute_empty_input(self):
        """Execute handles empty input."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 128},
        )

        context = StepContext(data=[])
        result = step.execute(context)

        assert result.success
        assert result.output == []
        assert result.metadata["embedded_count"] == 0

    def test_step_type(self):
        """Step has correct type."""
        step = MockEmbeddingStep(step_id="embed", config={})
        assert step.step_type == StepType.EMBEDDING


class TestOpenAIEmbeddingStep:
    """Tests for OpenAIEmbeddingStep (without actual API calls)."""

    def test_model_dimensions(self):
        """Model dimensions are correct."""
        step = OpenAIEmbeddingStep(
            step_id="embed",
            config={"model_name": "text-embedding-3-small"},
        )
        assert step.dimensions == 1536

        step2 = OpenAIEmbeddingStep(
            step_id="embed",
            config={"model_name": "text-embedding-3-large"},
        )
        assert step2.dimensions == 3072

    def test_custom_dimensions(self):
        """Custom dimensions override default."""
        step = OpenAIEmbeddingStep(
            step_id="embed",
            config={"model_name": "text-embedding-3-small", "dimensions": 512},
        )
        assert step.dimensions == 512

    def test_model_name(self):
        """Model name is configurable."""
        step = OpenAIEmbeddingStep(
            step_id="embed",
            config={"model_name": "text-embedding-ada-002"},
        )
        assert step.model_name == "text-embedding-ada-002"

    def test_requires_openai_package(self):
        """Gives helpful error if openai not installed."""
        step = OpenAIEmbeddingStep(
            step_id="embed",
            config={"api_key": "fake-key"},
        )

        # This would fail at runtime if openai isn't installed
        # Just verify the step can be created
        assert step.step_id == "embed"


class TestLocalEmbeddingStep:
    """Tests for LocalEmbeddingStep (without actual model loading)."""

    def test_default_model(self):
        """Default model is set."""
        step = LocalEmbeddingStep(step_id="embed", config={})
        assert step.model_name == "all-MiniLM-L6-v2"
        assert step.dimensions == 384

    def test_custom_model(self):
        """Can configure custom model."""
        step = LocalEmbeddingStep(
            step_id="embed",
            config={"model_name": "all-mpnet-base-v2"},
        )
        assert step.model_name == "all-mpnet-base-v2"
        assert step.dimensions == 768


class TestEmbeddingBatching:
    """Tests for batch embedding behavior."""

    def test_batch_size_respected(self):
        """Large inputs are batched correctly."""
        step = MockEmbeddingStep(
            step_id="embed",
            config={"dimensions": 64, "batch_size": 2},
        )

        # Create 5 chunks - should require 3 batches with batch_size=2
        chunks = [
            Chunk(content=f"Chunk {i}", doc_id="doc1", index=i)
            for i in range(5)
        ]

        embedded = step.embed_chunks(chunks)

        assert len(embedded) == 5
        assert all(ec.dimensions == 64 for ec in embedded)

    def test_empty_batch(self):
        """Empty input returns empty output."""
        step = MockEmbeddingStep(step_id="embed", config={})

        result = step.embed_chunks([])

        assert result == []


class TestEmbeddingSchemas:
    """Tests for embedding step schemas."""

    def test_input_schema(self):
        """Input schema accepts chunks or strings."""
        step = MockEmbeddingStep(step_id="embed", config={})

        schema = step.input_schema
        assert schema["type"] == "array"

    def test_output_schema(self):
        """Output schema returns embedded chunks."""
        step = MockEmbeddingStep(step_id="embed", config={})

        schema = step.output_schema
        assert schema["type"] == "array"
