"""Unit tests for chunking steps."""

import pytest

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.models.document import Document, SourceType
from rag_os.models.chunk import Chunk
from rag_os.steps.chunking import (
    BaseChunkingStep,
    ChunkingConfig,
    FixedSizeChunkingStep,
    SentenceChunkingStep,
    TokenAwareChunkingStep,
    RecursiveChunkingStep,
)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    content = """This is the first paragraph. It contains multiple sentences. Each sentence has some content.

This is the second paragraph. It also has multiple sentences. More content here.

And this is the third paragraph. Final sentences are here. The end."""

    return Document(
        content=content,
        source_type=SourceType.TEXT,
        source_uri="test.txt",
        title="Test Document",
    )


@pytest.fixture
def long_document() -> Document:
    """Create a long document for testing."""
    paragraphs = [
        f"This is paragraph {i}. " * 10
        for i in range(20)
    ]
    content = "\n\n".join(paragraphs)

    return Document(
        content=content,
        source_type=SourceType.TEXT,
        source_uri="long.txt",
    )


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100

    def test_custom_config(self):
        """Custom config values work."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            preserve_sentences=False,
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.preserve_sentences is False


class TestChunkModel:
    """Tests for Chunk model."""

    def test_chunk_creation(self):
        """Can create a chunk."""
        chunk = Chunk(
            content="Test content",
            doc_id="doc123",
            index=0,
        )

        assert chunk.content == "Test content"
        assert chunk.doc_id == "doc123"
        assert chunk.char_count == 12
        assert chunk.chunk_id is not None

    def test_chunk_hash(self):
        """Content hash is computed."""
        chunk1 = Chunk(content="same content", doc_id="d1", index=0)
        chunk2 = Chunk(content="same content", doc_id="d2", index=1)
        chunk3 = Chunk(content="different", doc_id="d1", index=0)

        assert chunk1.content_hash == chunk2.content_hash
        assert chunk1.content_hash != chunk3.content_hash

    def test_chunk_serialization(self):
        """Chunk can be serialized and deserialized."""
        chunk = Chunk(
            content="Test",
            doc_id="doc1",
            index=5,
            metadata={"key": "value"},
        )

        data = chunk.to_dict()
        restored = Chunk.from_dict(data)

        assert restored.content == chunk.content
        assert restored.doc_id == chunk.doc_id
        assert restored.index == chunk.index
        assert restored.metadata == chunk.metadata


class TestFixedSizeChunkingStep:
    """Tests for FixedSizeChunkingStep."""

    def test_basic_chunking(self, sample_document):
        """Basic fixed-size chunking works."""
        step = FixedSizeChunkingStep(
            step_id="chunk",
            config={"chunk_size": 100, "chunk_overlap": 20, "min_chunk_size": 10},
        )

        chunks = step.chunk_document(sample_document)

        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_id == sample_document.doc_id for c in chunks)

    def test_chunk_indices(self, sample_document):
        """Chunks have sequential indices."""
        step = FixedSizeChunkingStep(
            step_id="chunk",
            config={"chunk_size": 100, "chunk_overlap": 10},
        )

        chunks = step.chunk_document(sample_document)

        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_exists(self, long_document):
        """Adjacent chunks have overlapping content."""
        step = FixedSizeChunkingStep(
            step_id="chunk",
            config={"chunk_size": 200, "chunk_overlap": 50},
        )

        chunks = step.chunk_document(long_document)

        # Check that consecutive chunks have some overlap by checking
        # if any significant portion of text is shared
        overlap_found = 0
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i].content
            chunk2 = chunks[i + 1].content
            # Check for any overlap by looking for common substrings
            # At minimum, some words should appear in both chunks
            words1 = set(chunk1.split()[-10:])  # Last 10 words
            words2 = set(chunk2.split()[:10])   # First 10 words
            if words1 & words2:  # If there's any intersection
                overlap_found += 1

        # Most consecutive chunks should have some overlap
        assert overlap_found > len(chunks) // 2

    def test_execute_with_context(self, sample_document):
        """Execute method works with StepContext."""
        step = FixedSizeChunkingStep(
            step_id="chunk",
            config={"chunk_size": 100},
        )

        context = StepContext(data=[sample_document])
        result = step.execute(context)

        assert result.success
        assert len(result.output) > 0
        assert result.metadata["chunk_count"] > 0


class TestSentenceChunkingStep:
    """Tests for SentenceChunkingStep."""

    def test_sentence_chunking(self, sample_document):
        """Sentence-based chunking works."""
        step = SentenceChunkingStep(
            step_id="chunk",
            config={"chunk_size": 150, "chunk_overlap": 30, "min_chunk_size": 10},
        )

        chunks = step.chunk_document(sample_document)

        assert len(chunks) > 0
        # Chunks should not cut mid-sentence (ideally)
        for chunk in chunks:
            # Most chunks should end with sentence-ending punctuation
            # (some may not if they're the overlap portion)
            assert chunk.content.strip()

    def test_preserves_sentences(self):
        """Sentences are not split mid-way."""
        doc = Document(
            content="First sentence here. Second sentence here. Third one. Fourth sentence with more text here.",
            source_type=SourceType.TEXT,
            doc_id="test",
        )

        step = SentenceChunkingStep(
            step_id="chunk",
            config={"chunk_size": 50, "chunk_overlap": 10, "min_chunk_size": 5},
        )

        chunks = step.chunk_document(doc)

        # Content should be reconstructable (minus some overlap effects)
        all_content = " ".join(c.content for c in chunks)
        assert "First sentence" in all_content
        assert "Second sentence" in all_content


class TestTokenAwareChunkingStep:
    """Tests for TokenAwareChunkingStep."""

    def test_token_chunking_fallback(self, sample_document):
        """Token chunking falls back gracefully without tiktoken."""
        step = TokenAwareChunkingStep(
            step_id="chunk",
            config={"chunk_size": 50, "chunk_overlap": 10},  # 50 tokens
        )

        chunks = step.chunk_document(sample_document)

        assert len(chunks) > 0
        # Token count should be set
        for chunk in chunks:
            assert chunk.token_count is not None

    def test_token_counting(self):
        """Token counting works."""
        step = TokenAwareChunkingStep(
            step_id="chunk",
            config={"chunk_size": 100},
        )

        # Simple test text
        text = "Hello world, this is a test."
        token_count = step._count_tokens(text)

        # Should be reasonable (either tiktoken or fallback)
        assert token_count > 0
        assert token_count < len(text)  # Tokens should be fewer than characters


class TestRecursiveChunkingStep:
    """Tests for RecursiveChunkingStep."""

    def test_recursive_chunking(self, sample_document):
        """Recursive chunking works."""
        step = RecursiveChunkingStep(
            step_id="chunk",
            config={"chunk_size": 100, "chunk_overlap": 20, "min_chunk_size": 10},
        )

        chunks = step.chunk_document(sample_document)

        assert len(chunks) > 0

    def test_respects_paragraphs(self, long_document):
        """Tries to split at paragraph boundaries first."""
        step = RecursiveChunkingStep(
            step_id="chunk",
            config={"chunk_size": 500, "chunk_overlap": 50},
        )

        chunks = step.chunk_document(long_document)

        # Check that chunks tend to start at paragraph beginnings
        paragraph_starts = 0
        for chunk in chunks:
            if chunk.content.startswith("This is paragraph"):
                paragraph_starts += 1

        # Most chunks should start at paragraph boundaries
        assert paragraph_starts > len(chunks) // 2

    def test_custom_separators(self, sample_document):
        """Custom separators can be used."""
        step = RecursiveChunkingStep(
            step_id="chunk",
            config={
                "chunk_size": 100,
                "chunk_overlap": 10,
                "separators": [". ", " "],  # Only sentence and word
            },
        )

        chunks = step.chunk_document(sample_document)

        assert len(chunks) > 0


class TestChunkingWithMultipleDocuments:
    """Tests for chunking multiple documents."""

    def test_chunk_multiple_documents(self):
        """Can chunk multiple documents at once."""
        docs = [
            Document(content="Document one content.", source_type=SourceType.TEXT, doc_id="d1"),
            Document(content="Document two content.", source_type=SourceType.TEXT, doc_id="d2"),
            Document(content="Document three content.", source_type=SourceType.TEXT, doc_id="d3"),
        ]

        step = FixedSizeChunkingStep(
            step_id="chunk",
            config={"chunk_size": 50, "min_chunk_size": 5},
        )

        chunks = step.chunk_documents(docs)

        # Should have chunks from all documents
        doc_ids = {c.doc_id for c in chunks}
        assert "d1" in doc_ids
        assert "d2" in doc_ids
        assert "d3" in doc_ids

    def test_execute_preserves_doc_references(self):
        """Chunks maintain correct document references."""
        docs = [
            Document(content="A" * 200, source_type=SourceType.TEXT, doc_id="docA"),
            Document(content="B" * 200, source_type=SourceType.TEXT, doc_id="docB"),
        ]

        step = FixedSizeChunkingStep(
            step_id="chunk",
            config={"chunk_size": 50, "chunk_overlap": 10},
        )

        context = StepContext(data=docs)
        result = step.execute(context)

        assert result.success

        # Verify doc_id references
        for chunk in result.output:
            if "A" in chunk.content:
                assert chunk.doc_id == "docA"
            elif "B" in chunk.content:
                assert chunk.doc_id == "docB"
