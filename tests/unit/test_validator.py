"""Unit tests for pipeline validator."""

import pytest

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.spec import StepSpec, PipelineSpec
from rag_os.core.registry import StepRegistry
from rag_os.core.validator import (
    PipelineValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_pipeline,
)


# Test step implementations
class MockIngestionStep(Step):
    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"source": {"type": "string"}}}

    @property
    def output_schema(self) -> dict:
        return {"type": "array", "items": {"type": "object"}}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([])


class MockChunkingStep(Step):
    @property
    def input_schema(self) -> dict:
        return {"type": "array"}

    @property
    def output_schema(self) -> dict:
        return {"type": "array"}

    @property
    def config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"chunk_size": {"type": "integer"}},
            "required": ["chunk_size"],
        }

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([])


class MockEmbeddingStep(Step):
    @property
    def input_schema(self) -> dict:
        return {"type": "array"}

    @property
    def output_schema(self) -> dict:
        return {"type": "array"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([])


class MockRetrievalStep(Step):
    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    @property
    def output_schema(self) -> dict:
        return {"type": "array"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok([])


class MockLLMStep(Step):
    @property
    def input_schema(self) -> dict:
        return {"type": "object"}

    @property
    def output_schema(self) -> dict:
        return {"type": "string"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok("")


class MockLLMStepFallback(Step):
    @property
    def input_schema(self) -> dict:
        return {"type": "object"}

    @property
    def output_schema(self) -> dict:
        return {"type": "string"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok("")


@pytest.fixture
def registry():
    """Provide a registry with test steps."""
    reg = StepRegistry()
    reg.clear()

    reg.register(MockIngestionStep, "TestIngestionStep", StepType.INGESTION)
    reg.register(MockChunkingStep, "TestChunkingStep", StepType.CHUNKING)
    reg.register(MockEmbeddingStep, "TestEmbeddingStep", StepType.EMBEDDING)
    reg.register(MockRetrievalStep, "TestRetrievalStep", StepType.RETRIEVAL)
    reg.register(MockLLMStep, "TestLLMStep", StepType.LLM_EXECUTION)
    reg.register(MockLLMStepFallback, "TestLLMStepFallback", StepType.LLM_EXECUTION)

    yield reg
    reg.clear()


@pytest.fixture
def validator(registry):
    """Provide a validator with the test registry."""
    return PipelineValidator(registry)


class TestValidationIssue:
    """Tests for ValidationIssue."""

    def test_basic_issue(self):
        """ValidationIssue can be created and stringified."""
        issue = ValidationIssue(
            message="Something is wrong",
            severity=ValidationSeverity.ERROR,
        )

        assert "[ERROR]" in str(issue)
        assert "Something is wrong" in str(issue)

    def test_issue_with_step(self):
        """ValidationIssue includes step info in string."""
        issue = ValidationIssue(
            message="Invalid config",
            severity=ValidationSeverity.WARNING,
            step_id="my_step",
            field="config",
        )

        issue_str = str(issue)
        assert "[WARNING]" in issue_str
        assert "my_step" in issue_str
        assert "config" in issue_str

    def test_issue_with_suggestion(self):
        """ValidationIssue includes suggestion in string."""
        issue = ValidationIssue(
            message="Missing field",
            severity=ValidationSeverity.ERROR,
            suggestion="Add the required field",
        )

        assert "Suggestion:" in str(issue)
        assert "Add the required field" in str(issue)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_empty_result_is_valid(self):
        """Empty result defaults to valid."""
        result = ValidationResult(valid=True)
        assert result.valid
        assert len(result.issues) == 0

    def test_add_error_invalidates(self):
        """Adding an error makes result invalid."""
        result = ValidationResult(valid=True)
        result.add_error("Something broke")

        assert not result.valid
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Adding a warning doesn't invalidate."""
        result = ValidationResult(valid=True)
        result.add_warning("Consider changing this")

        assert result.valid
        assert len(result.warnings) == 1

    def test_filter_by_severity(self):
        """Issues can be filtered by severity."""
        result = ValidationResult(valid=True)
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.add_info("Info 1")

        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert len(result.info) == 1
        assert len(result.issues) == 4

    def test_str_representation(self):
        """ValidationResult has useful string repr."""
        result = ValidationResult(valid=True)
        result.add_warning("A warning")

        result_str = str(result)
        assert "VALID" in result_str
        assert "Warnings: 1" in result_str


class TestPipelineValidator:
    """Tests for PipelineValidator."""

    def test_valid_pipeline(self, validator):
        """Valid pipeline passes validation."""
        spec = PipelineSpec(
            name="valid-pipeline",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="ingest",
                    step_type=StepType.INGESTION,
                    step_class="TestIngestionStep",
                ),
                StepSpec(
                    step_id="chunk",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={"chunk_size": 512},
                    dependencies=["ingest"],
                ),
                StepSpec(
                    step_id="retrieve",
                    step_type=StepType.RETRIEVAL,
                    step_class="TestRetrievalStep",
                    dependencies=["chunk"],
                ),
            ],
        )

        result = validator.validate(spec)
        assert result.valid, f"Validation failed: {result}"

    def test_missing_step_class(self, validator):
        """Missing step class is an error."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="bad_step",
                    step_type=StepType.INGESTION,
                    step_class="NonExistentStep",
                ),
            ],
        )

        result = validator.validate(spec)
        assert not result.valid
        assert any("not registered" in e.message for e in result.errors)

    def test_invalid_config(self, validator):
        """Invalid step config is an error."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="chunk",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={},  # Missing required chunk_size
                ),
            ],
        )

        result = validator.validate(spec)
        assert not result.valid
        assert any("chunk_size" in e.message for e in result.errors)

    def test_dependency_order(self, validator):
        """Dependencies must come before dependent steps."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="chunk",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={"chunk_size": 512},
                    dependencies=["ingest"],  # Depends on step that comes after
                ),
                StepSpec(
                    step_id="ingest",
                    step_type=StepType.INGESTION,
                    step_class="TestIngestionStep",
                ),
            ],
        )

        result = validator.validate(spec)
        assert not result.valid
        assert any("must come before" in e.message for e in result.errors)

    def test_circular_dependency_detected(self, validator):
        """Circular dependencies are detected."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="step_a",
                    step_type=StepType.INGESTION,
                    step_class="TestIngestionStep",
                    dependencies=["step_c"],
                ),
                StepSpec(
                    step_id="step_b",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={"chunk_size": 512},
                    dependencies=["step_a"],
                ),
                StepSpec(
                    step_id="step_c",
                    step_type=StepType.EMBEDDING,
                    step_class="TestEmbeddingStep",
                    dependencies=["step_b"],
                ),
            ],
        )

        result = validator.validate(spec)
        assert not result.valid
        assert any("Circular dependency" in e.message for e in result.errors)

    def test_schema_type_mismatch_warning(self, validator):
        """Schema type mismatch produces a warning."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="llm",
                    step_type=StepType.LLM_EXECUTION,
                    step_class="TestLLMStep",
                ),
                StepSpec(
                    step_id="chunk",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={"chunk_size": 512},
                    dependencies=["llm"],  # LLM outputs string, chunking expects array
                ),
            ],
        )

        result = validator.validate(spec)
        # Should have a warning about schema mismatch
        assert any("mismatch" in w.message.lower() for w in result.warnings)

    def test_fallback_type_mismatch_warning(self, validator):
        """Fallback of different type produces warning."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="ingest",
                    step_type=StepType.INGESTION,
                    step_class="TestIngestionStep",
                    fallback_step="chunk",  # Different type
                ),
                StepSpec(
                    step_id="chunk",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={"chunk_size": 512},
                ),
            ],
        )

        result = validator.validate(spec)
        assert any("different type" in w.message for w in result.warnings)

    def test_enabled_fallback_info(self, validator):
        """Enabled fallback step produces info message."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="llm_main",
                    step_type=StepType.LLM_EXECUTION,
                    step_class="TestLLMStep",
                    fallback_step="llm_fallback",
                ),
                StepSpec(
                    step_id="llm_fallback",
                    step_type=StepType.LLM_EXECUTION,
                    step_class="TestLLMStepFallback",
                    enabled=True,  # Fallback is enabled
                ),
            ],
        )

        result = validator.validate(spec)
        assert any("enabled" in i.message for i in result.info)

    def test_no_retrieval_step_info(self, validator):
        """Pipeline without retrieval produces info."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="ingest",
                    step_type=StepType.INGESTION,
                    step_class="TestIngestionStep",
                ),
                StepSpec(
                    step_id="chunk",
                    step_type=StepType.CHUNKING,
                    step_class="TestChunkingStep",
                    config={"chunk_size": 512},
                ),
            ],
        )

        result = validator.validate(spec)
        assert any("retrieval" in i.message.lower() for i in result.info)

    def test_all_steps_disabled_error(self, validator):
        """Pipeline with all steps disabled is an error."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="ingest",
                    step_type=StepType.INGESTION,
                    step_class="TestIngestionStep",
                    enabled=False,
                ),
            ],
        )

        result = validator.validate(spec)
        assert not result.valid
        assert any("no enabled steps" in e.message for e in result.errors)


class TestValidatePipelineFunction:
    """Tests for validate_pipeline convenience function."""

    def test_convenience_function(self, registry):
        """validate_pipeline function works with registry."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="retrieve",
                    step_type=StepType.RETRIEVAL,
                    step_class="TestRetrievalStep",
                ),
            ],
        )

        result = validate_pipeline(spec, registry)
        assert result.valid

    def test_convenience_function_detects_errors(self, registry):
        """validate_pipeline function detects errors."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="bad",
                    step_type=StepType.INGESTION,
                    step_class="MissingStep",
                ),
            ],
        )

        result = validate_pipeline(spec, registry)
        assert not result.valid
