"""Unit tests for step registry."""

import pytest

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.registry import (
    StepRegistry,
    StepMetadata,
    register_step,
    get_registry,
)


# Test step implementations
class MockIngestionStep(Step):
    """Mock ingestion step for testing."""

    @property
    def input_schema(self) -> dict:
        return {"type": "object"}

    @property
    def output_schema(self) -> dict:
        return {"type": "array"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([{"doc": "test"}])


class MockChunkingStep(Step):
    """Mock chunking step for testing."""

    @property
    def input_schema(self) -> dict:
        return {"type": "array"}

    @property
    def output_schema(self) -> dict:
        return {"type": "array"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([{"chunk": "test"}])


class MockEmbeddingStep(Step):
    """Mock embedding step for testing."""

    @property
    def input_schema(self) -> dict:
        return {"type": "array"}

    @property
    def output_schema(self) -> dict:
        return {"type": "array"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([{"vector": [0.1, 0.2]}])


@pytest.fixture
def registry():
    """Provide a clean registry for each test."""
    reg = StepRegistry()
    reg.clear()
    yield reg
    reg.clear()


class TestStepMetadata:
    """Tests for StepMetadata."""

    def test_metadata_creation(self):
        """StepMetadata can be created with all fields."""
        metadata = StepMetadata(
            step_class=MockIngestionStep,
            name="test_ingestor",
            step_type=StepType.INGESTION,
            description="A test ingestion step",
            version="1.2.3",
            author="Test Author",
            tags=["test", "mock"],
        )

        assert metadata.step_class == MockIngestionStep
        assert metadata.name == "test_ingestor"
        assert metadata.step_type == StepType.INGESTION
        assert metadata.description == "A test ingestion step"
        assert metadata.version == "1.2.3"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "mock"]

    def test_metadata_defaults(self):
        """StepMetadata has sensible defaults."""
        metadata = StepMetadata(
            step_class=MockIngestionStep,
            name="test",
            step_type=StepType.INGESTION,
        )

        assert metadata.description == ""
        assert metadata.version == "1.0.0"
        assert metadata.author == ""
        assert metadata.tags == []

    def test_metadata_repr(self):
        """StepMetadata has useful repr."""
        metadata = StepMetadata(
            step_class=MockIngestionStep,
            name="my_step",
            step_type=StepType.CHUNKING,
            version="2.0.0",
        )

        repr_str = repr(metadata)
        assert "my_step" in repr_str
        assert "chunking" in repr_str.lower()
        assert "2.0.0" in repr_str


class TestStepRegistry:
    """Tests for StepRegistry."""

    def test_singleton(self):
        """StepRegistry is a singleton."""
        reg1 = StepRegistry()
        reg2 = StepRegistry()
        assert reg1 is reg2

    def test_register_step(self, registry):
        """Steps can be registered."""
        registry.register(
            step_class=MockIngestionStep,
            name="file_ingestor",
            step_type=StepType.INGESTION,
            description="Ingests files",
        )

        assert registry.has("file_ingestor")
        assert len(registry) == 1

    def test_register_multiple_steps(self, registry):
        """Multiple steps can be registered."""
        registry.register(MockIngestionStep, "ingestor", StepType.INGESTION)
        registry.register(MockChunkingStep, "chunker", StepType.CHUNKING)
        registry.register(MockEmbeddingStep, "embedder", StepType.EMBEDDING)

        assert len(registry) == 3
        assert "ingestor" in registry
        assert "chunker" in registry
        assert "embedder" in registry

    def test_register_duplicate_rejected(self, registry):
        """Duplicate names are rejected."""
        registry.register(MockIngestionStep, "my_step", StepType.INGESTION)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockChunkingStep, "my_step", StepType.CHUNKING)

    def test_register_non_step_rejected(self, registry):
        """Non-Step classes are rejected."""
        with pytest.raises(ValueError, match="subclass of Step"):
            registry.register(str, "invalid", StepType.INGESTION)  # type: ignore

    def test_get_step_class(self, registry):
        """get() returns the step class."""
        registry.register(MockIngestionStep, "my_ingestor", StepType.INGESTION)

        step_class = registry.get("my_ingestor")
        assert step_class is MockIngestionStep

    def test_get_nonexistent_returns_none(self, registry):
        """get() returns None for unregistered steps."""
        assert registry.get("nonexistent") is None

    def test_get_metadata(self, registry):
        """get_metadata() returns full metadata."""
        registry.register(
            MockIngestionStep,
            "detailed_step",
            StepType.INGESTION,
            description="Detailed description",
            version="2.0.0",
            author="Test",
            tags=["tag1", "tag2"],
        )

        metadata = registry.get_metadata("detailed_step")
        assert metadata is not None
        assert metadata.description == "Detailed description"
        assert metadata.version == "2.0.0"
        assert metadata.tags == ["tag1", "tag2"]

    def test_get_metadata_nonexistent(self, registry):
        """get_metadata() returns None for unregistered steps."""
        assert registry.get_metadata("nonexistent") is None

    def test_get_by_type(self, registry):
        """get_by_type() filters by step type."""
        registry.register(MockIngestionStep, "ingestor1", StepType.INGESTION)
        registry.register(MockIngestionStep, "ingestor2", StepType.INGESTION)
        registry.register(MockChunkingStep, "chunker1", StepType.CHUNKING)

        ingestion_steps = registry.get_by_type(StepType.INGESTION)
        assert len(ingestion_steps) == 2
        assert "ingestor1" in ingestion_steps
        assert "ingestor2" in ingestion_steps

        chunking_steps = registry.get_by_type(StepType.CHUNKING)
        assert len(chunking_steps) == 1
        assert "chunker1" in chunking_steps

        embedding_steps = registry.get_by_type(StepType.EMBEDDING)
        assert len(embedding_steps) == 0

    def test_list_steps(self, registry):
        """list_steps() returns all registered names."""
        registry.register(MockIngestionStep, "step_a", StepType.INGESTION)
        registry.register(MockChunkingStep, "step_b", StepType.CHUNKING)

        names = registry.list_steps()
        assert set(names) == {"step_a", "step_b"}

    def test_list_metadata(self, registry):
        """list_metadata() returns all metadata."""
        registry.register(MockIngestionStep, "step_a", StepType.INGESTION)
        registry.register(MockChunkingStep, "step_b", StepType.CHUNKING)

        all_metadata = registry.list_metadata()
        assert len(all_metadata) == 2
        names = {m.name for m in all_metadata}
        assert names == {"step_a", "step_b"}

    def test_unregister(self, registry):
        """Steps can be unregistered."""
        registry.register(MockIngestionStep, "temp_step", StepType.INGESTION)
        assert registry.has("temp_step")

        result = registry.unregister("temp_step")
        assert result is True
        assert not registry.has("temp_step")
        assert len(registry.get_by_type(StepType.INGESTION)) == 0

    def test_unregister_nonexistent(self, registry):
        """Unregistering nonexistent step returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_create_step(self, registry):
        """create_step() instantiates a registered step."""
        registry.register(MockChunkingStep, "my_chunker", StepType.CHUNKING)

        step = registry.create_step(
            name="my_chunker",
            step_id="chunk_step_1",
            config={"chunk_size": 512},
        )

        assert isinstance(step, MockChunkingStep)
        assert step.step_id == "chunk_step_1"
        assert step.step_type == StepType.CHUNKING
        assert step.config == {"chunk_size": 512}

    def test_create_step_not_registered(self, registry):
        """create_step() raises KeyError for unregistered steps."""
        with pytest.raises(KeyError, match="not registered"):
            registry.create_step("nonexistent", "step_1")

    def test_clear(self, registry):
        """clear() removes all registered steps."""
        registry.register(MockIngestionStep, "step1", StepType.INGESTION)
        registry.register(MockChunkingStep, "step2", StepType.CHUNKING)
        assert len(registry) == 2

        registry.clear()
        assert len(registry) == 0
        assert registry.list_steps() == []

    def test_contains(self, registry):
        """__contains__ works for membership testing."""
        registry.register(MockIngestionStep, "my_step", StepType.INGESTION)

        assert "my_step" in registry
        assert "other_step" not in registry


class TestRegisterStepDecorator:
    """Tests for @register_step decorator."""

    def test_decorator_registers_step(self, registry):
        """@register_step decorator registers the class."""

        @register_step(
            name="decorated_step",
            step_type=StepType.RETRIEVAL,
            description="A decorated step",
        )
        class DecoratedRetrievalStep(Step):
            @property
            def input_schema(self) -> dict:
                return {}

            @property
            def output_schema(self) -> dict:
                return {}

            def execute(self, context: StepContext) -> StepResult:
                return StepResult.ok([])

        assert registry.has("decorated_step")
        assert registry.get("decorated_step") is DecoratedRetrievalStep
        metadata = registry.get_metadata("decorated_step")
        assert metadata.description == "A decorated step"

    def test_decorator_with_all_options(self, registry):
        """Decorator accepts all metadata options."""

        @register_step(
            name="full_options_step",
            step_type=StepType.RERANKING,
            description="Full options",
            version="3.0.0",
            author="Decorator Test",
            tags=["decorated", "test"],
        )
        class FullOptionsStep(Step):
            @property
            def input_schema(self) -> dict:
                return {}

            @property
            def output_schema(self) -> dict:
                return {}

            def execute(self, context: StepContext) -> StepResult:
                return StepResult.ok(None)

        metadata = registry.get_metadata("full_options_step")
        assert metadata.version == "3.0.0"
        assert metadata.author == "Decorator Test"
        assert metadata.tags == ["decorated", "test"]

    def test_decorator_preserves_class(self, registry):
        """Decorator returns the original class unchanged."""

        @register_step(name="preserved", step_type=StepType.EMBEDDING)
        class PreservedStep(Step):
            custom_attr = "preserved"

            @property
            def input_schema(self) -> dict:
                return {}

            @property
            def output_schema(self) -> dict:
                return {}

            def execute(self, context: StepContext) -> StepResult:
                return StepResult.ok(None)

        assert PreservedStep.custom_attr == "preserved"
        assert PreservedStep.__name__ == "PreservedStep"


class TestGetRegistry:
    """Tests for get_registry() helper."""

    def test_get_registry_returns_singleton(self, registry):
        """get_registry() returns the singleton."""
        reg = get_registry()
        assert reg is registry
