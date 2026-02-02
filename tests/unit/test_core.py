"""Unit tests for core components."""

import pytest
from datetime import datetime, timezone

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step


class TestStepType:
    """Tests for StepType enum."""

    def test_all_step_types_exist(self):
        """Verify all expected step types are defined."""
        expected_types = [
            "ingestion",
            "chunking",
            "embedding",
            "indexing",
            "retrieval",
            "reranking",
            "prompt_assembly",
            "llm_execution",
            "post_processing",
        ]
        actual_types = [t.value for t in StepType]
        assert set(expected_types) == set(actual_types)

    def test_step_type_string_conversion(self):
        """StepType should convert to string properly."""
        assert str(StepType.INGESTION) == "ingestion"
        assert str(StepType.CHUNKING) == "chunking"


class TestStepContext:
    """Tests for StepContext dataclass."""

    def test_default_creation(self):
        """Context can be created with defaults."""
        ctx = StepContext()
        assert ctx.data is None
        assert ctx.metadata == {}
        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 36  # UUID format
        assert ctx.step_outputs == {}

    def test_with_data(self):
        """with_data creates new context with updated data."""
        ctx = StepContext(data="original", metadata={"key": "value"})
        new_ctx = ctx.with_data("updated")

        assert new_ctx.data == "updated"
        assert ctx.data == "original"  # Original unchanged
        assert new_ctx.metadata == {"key": "value"}  # Preserved
        assert new_ctx.trace_id == ctx.trace_id  # Preserved

    def test_with_step_output(self):
        """with_step_output records step outputs."""
        ctx = StepContext()
        ctx = ctx.with_step_output("step1", {"result": "data1"})
        ctx = ctx.with_step_output("step2", {"result": "data2"})

        assert ctx.get_step_output("step1") == {"result": "data1"}
        assert ctx.get_step_output("step2") == {"result": "data2"}
        assert ctx.get_step_output("nonexistent") is None

    def test_update_metadata(self):
        """update_metadata adds to existing metadata."""
        ctx = StepContext(metadata={"existing": "value"})
        new_ctx = ctx.update_metadata(new_key="new_value")

        assert new_ctx.metadata == {"existing": "value", "new_key": "new_value"}
        assert ctx.metadata == {"existing": "value"}  # Original unchanged

    def test_immutability(self):
        """Context operations should not mutate original."""
        ctx = StepContext(data="original", step_outputs={"step1": "out1"})

        # These should create new instances
        ctx.with_data("new")
        ctx.with_step_output("step2", "out2")
        ctx.update_metadata(key="value")

        # Original should be unchanged
        assert ctx.data == "original"
        assert "step2" not in ctx.step_outputs
        assert "key" not in ctx.metadata


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_default_success(self):
        """Default result is successful."""
        result = StepResult(output="data")
        assert result.success is True
        assert result.output == "data"
        assert result.error is None

    def test_ok_factory(self):
        """StepResult.ok() creates successful result."""
        result = StepResult.ok({"key": "value"}, extra_info="metadata")

        assert result.success is True
        assert result.output == {"key": "value"}
        assert result.metadata == {"extra_info": "metadata"}

    def test_fail_factory(self):
        """StepResult.fail() creates failed result."""
        result = StepResult.fail("Something went wrong", step="test_step")

        assert result.success is False
        assert result.output is None
        assert result.error == "Something went wrong"
        assert result.metadata == {"step": "test_step"}

    def test_with_latency(self):
        """with_latency adds timing information."""
        result = StepResult.ok("data")
        timed = result.with_latency(150.5)

        assert timed.latency_ms == 150.5
        assert result.latency_ms == 0.0  # Original unchanged

    def test_with_token_usage(self):
        """with_token_usage adds token information."""
        result = StepResult.ok("response")
        with_tokens = result.with_token_usage(
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert with_tokens.token_usage["prompt_tokens"] == 100
        assert with_tokens.token_usage["completion_tokens"] == 50
        assert with_tokens.token_usage["total_tokens"] == 150

    def test_created_at_timestamp(self):
        """Result should have creation timestamp."""
        before = datetime.now(timezone.utc)
        result = StepResult.ok("data")
        after = datetime.now(timezone.utc)

        assert before <= result.created_at <= after


class DummyStep(Step):
    """A concrete implementation of Step for testing."""

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    @property
    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"processed": {"type": "string"}},
            "required": ["processed"],
        }

    @property
    def config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"uppercase": {"type": "boolean"}},
            "required": ["uppercase"],
        }

    def execute(self, context: StepContext) -> StepResult:
        text = context.data.get("text", "")
        if self.config.get("uppercase"):
            text = text.upper()
        return StepResult.ok({"processed": text})


class TestStep:
    """Tests for Step abstract base class."""

    def test_step_creation(self):
        """Step can be instantiated with required fields."""
        step = DummyStep(
            step_id="test_step_1",
            step_type=StepType.POST_PROCESSING,
            config={"uppercase": True},
        )

        assert step.step_id == "test_step_1"
        assert step.step_type == StepType.POST_PROCESSING
        assert step.config == {"uppercase": True}

    def test_step_execution(self):
        """Step executes and returns result."""
        step = DummyStep(
            step_id="test",
            step_type=StepType.POST_PROCESSING,
            config={"uppercase": True},
        )
        context = StepContext(data={"text": "hello"})

        result = step.execute(context)

        assert result.success is True
        assert result.output == {"processed": "HELLO"}

    def test_validate_input_success(self):
        """Input validation passes with valid data."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})
        context = StepContext(data={"text": "hello"})

        errors = step.validate_input(context)
        assert errors == []

    def test_validate_input_missing_required(self):
        """Input validation fails with missing required field."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})
        context = StepContext(data={"other": "field"})

        errors = step.validate_input(context)
        assert len(errors) == 1
        assert "text" in errors[0]

    def test_validate_input_none_data(self):
        """Input validation fails when data is None but fields required."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})
        context = StepContext(data=None)

        errors = step.validate_input(context)
        assert len(errors) == 1

    def test_validate_output_success(self):
        """Output validation passes with valid result."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})
        result = StepResult.ok({"processed": "HELLO"})

        errors = step.validate_output(result)
        assert errors == []

    def test_validate_output_missing_required(self):
        """Output validation fails with missing required field."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})
        result = StepResult.ok({"other": "field"})

        errors = step.validate_output(result)
        assert len(errors) == 1
        assert "processed" in errors[0]

    def test_validate_output_skipped_on_failure(self):
        """Output validation is skipped for failed results."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})
        result = StepResult.fail("Something broke")

        errors = step.validate_output(result)
        assert errors == []

    def test_validate_config_success(self):
        """Config validation passes with valid config."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})

        errors = step.validate_config()
        assert errors == []

    def test_validate_config_missing_required(self):
        """Config validation fails with missing required field."""
        step = DummyStep("test", StepType.POST_PROCESSING, {})

        errors = step.validate_config()
        assert len(errors) == 1
        assert "uppercase" in errors[0]

    def test_runtime_requirements_default(self):
        """Default runtime requirements is empty dict."""
        step = DummyStep("test", StepType.POST_PROCESSING, {"uppercase": True})

        assert step.runtime_requirements == {}

    def test_step_repr(self):
        """Step has useful string representation."""
        step = DummyStep("my_step", StepType.CHUNKING, {})

        repr_str = repr(step)
        assert "DummyStep" in repr_str
        assert "my_step" in repr_str
        assert "chunking" in repr_str.lower()
