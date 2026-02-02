"""Unit tests for pipeline executor."""

import pytest
import time

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.spec import StepSpec, PipelineSpec
from rag_os.core.registry import StepRegistry
from rag_os.core.runtime import RuntimeContext, ExecutionMode, FailureMode
from rag_os.core.executor import (
    PipelineExecutor,
    PipelineResult,
    StepExecutionResult,
    execute_pipeline,
)


# Test step implementations
class AddOneStep(Step):
    """Adds 1 to input number."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok(context.data + 1)


class MultiplyStep(Step):
    """Multiplies input by configured factor."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    @property
    def config_schema(self) -> dict:
        return {"type": "object", "properties": {"factor": {"type": "number"}}}

    def execute(self, context: StepContext) -> StepResult:
        factor = self.config.get("factor", 2)
        return StepResult.ok(context.data * factor)


class FailingStep(Step):
    """Always fails."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.fail("Intentional failure")


class ExceptionStep(Step):
    """Raises an exception."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    def execute(self, context: StepContext) -> StepResult:
        raise ValueError("Intentional exception")


class SlowStep(Step):
    """Takes some time to execute."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    def execute(self, context: StepContext) -> StepResult:
        delay = self.config.get("delay_ms", 10) / 1000
        time.sleep(delay)
        return StepResult.ok(context.data)


class TokenUsingStep(Step):
    """Reports token usage."""

    @property
    def input_schema(self) -> dict:
        return {"type": "string"}

    @property
    def output_schema(self) -> dict:
        return {"type": "string"}

    def execute(self, context: StepContext) -> StepResult:
        result = StepResult.ok(f"Processed: {context.data}")
        return result.with_token_usage(prompt_tokens=100, completion_tokens=50)


class FallbackStep(Step):
    """Fallback that always succeeds."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    def execute(self, context: StepContext) -> StepResult:
        return StepResult.ok(context.data * -1)  # Negate as fallback


@pytest.fixture
def registry():
    """Provide a registry with test steps."""
    reg = StepRegistry()
    reg.clear()

    reg.register(AddOneStep, "AddOneStep", StepType.POST_PROCESSING)
    reg.register(MultiplyStep, "MultiplyStep", StepType.POST_PROCESSING)
    reg.register(FailingStep, "FailingStep", StepType.POST_PROCESSING)
    reg.register(ExceptionStep, "ExceptionStep", StepType.POST_PROCESSING)
    reg.register(SlowStep, "SlowStep", StepType.POST_PROCESSING)
    reg.register(TokenUsingStep, "TokenUsingStep", StepType.LLM_EXECUTION)
    reg.register(FallbackStep, "FallbackStep", StepType.POST_PROCESSING)

    yield reg
    reg.clear()


@pytest.fixture
def executor(registry):
    """Provide an executor with the test registry."""
    return PipelineExecutor(registry)


class TestStepExecutionResult:
    """Tests for StepExecutionResult."""

    def test_basic_result(self):
        """Can create step execution result."""
        result = StepExecutionResult(
            step_id="step1",
            step_type="post_processing",
            success=True,
            output=42,
            latency_ms=100.5,
        )

        assert result.step_id == "step1"
        assert result.success
        assert result.output == 42
        assert result.latency_ms == 100.5


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_get_step_result(self):
        """Can get specific step result."""
        result = PipelineResult(
            success=True,
            step_results=[
                StepExecutionResult("s1", "type", True, output=1),
                StepExecutionResult("s2", "type", True, output=2),
            ],
        )

        assert result.get_step_result("s1").output == 1
        assert result.get_step_result("s2").output == 2
        assert result.get_step_result("s3") is None

    def test_filter_results(self):
        """Can filter results by status."""
        result = PipelineResult(
            success=False,
            step_results=[
                StepExecutionResult("s1", "type", True, output=1),
                StepExecutionResult("s2", "type", False, error="failed"),
                StepExecutionResult("s3", "type", True, output=3, skipped=True, skip_reason="budget"),
            ],
        )

        assert len(result.successful_steps) == 1
        assert len(result.failed_steps) == 1
        assert len(result.skipped_steps) == 1


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""

    def test_simple_pipeline(self, executor):
        """Execute simple single-step pipeline."""
        spec = PipelineSpec(
            name="simple",
            version="1.0.0",
            steps=[
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert result.success
        assert result.output == 6
        assert len(result.step_results) == 1
        assert result.step_results[0].success

    def test_multi_step_pipeline(self, executor):
        """Execute multi-step pipeline."""
        spec = PipelineSpec(
            name="multi",
            version="1.0.0",
            steps=[
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
                StepSpec(
                    step_id="multiply",
                    step_type=StepType.POST_PROCESSING,
                    step_class="MultiplyStep",
                    config={"factor": 3},
                ),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert result.success
        assert result.output == 18  # (5 + 1) * 3
        assert len(result.step_results) == 2

    def test_step_failure_fail_fast(self, executor):
        """Fail fast mode stops on first failure."""
        spec = PipelineSpec(
            name="failing",
            version="1.0.0",
            steps=[
                StepSpec(step_id="fail", step_type=StepType.POST_PROCESSING, step_class="FailingStep"),
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            ],
        )

        runtime = RuntimeContext.create(failure_mode=FailureMode.FAIL_FAST)
        result = executor.execute(spec, runtime, initial_data=5)

        assert not result.success
        assert "Intentional failure" in result.error
        assert len(result.step_results) == 1  # Only first step executed

    def test_step_failure_graceful(self, executor):
        """Graceful mode continues after failure."""
        spec = PipelineSpec(
            name="graceful",
            version="1.0.0",
            steps=[
                StepSpec(step_id="fail", step_type=StepType.POST_PROCESSING, step_class="FailingStep"),
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            ],
        )

        runtime = RuntimeContext.create(failure_mode=FailureMode.GRACEFUL)
        result = executor.execute(spec, runtime, initial_data=5)

        assert not result.success  # Overall still fails
        assert len(result.step_results) == 2  # Both steps executed
        assert len(result.failed_steps) == 1

    def test_step_exception(self, executor):
        """Handles step exceptions."""
        spec = PipelineSpec(
            name="exception",
            version="1.0.0",
            steps=[
                StepSpec(step_id="exc", step_type=StepType.POST_PROCESSING, step_class="ExceptionStep"),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert not result.success
        assert "Intentional exception" in result.step_results[0].error

    def test_missing_step_class(self, executor):
        """Handles missing step class."""
        spec = PipelineSpec(
            name="missing",
            version="1.0.0",
            steps=[
                StepSpec(step_id="missing", step_type=StepType.POST_PROCESSING, step_class="NonExistent"),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert not result.success
        assert "not found" in result.step_results[0].error

    def test_runtime_overrides(self, executor):
        """Runtime overrides are applied."""
        spec = PipelineSpec(
            name="override",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="multiply",
                    step_type=StepType.POST_PROCESSING,
                    step_class="MultiplyStep",
                    config={"factor": 2},
                ),
            ],
        )

        runtime = RuntimeContext.create(
            overrides={"multiply": {"factor": 10}}
        )
        result = executor.execute(spec, runtime, initial_data=5)

        assert result.success
        assert result.output == 50  # 5 * 10 (override), not 5 * 2

    def test_dry_run(self, executor):
        """Dry run skips actual execution."""
        spec = PipelineSpec(
            name="dryrun",
            version="1.0.0",
            steps=[
                StepSpec(step_id="fail", step_type=StepType.POST_PROCESSING, step_class="FailingStep"),
            ],
        )

        runtime = RuntimeContext.create(execution_mode=ExecutionMode.DRY_RUN)
        result = executor.execute(spec, runtime, initial_data=5)

        assert result.success  # Doesn't actually fail in dry run
        assert result.output == 5  # Data passed through unchanged

    def test_token_budget_tracking(self, executor):
        """Token usage is tracked."""
        spec = PipelineSpec(
            name="tokens",
            version="1.0.0",
            steps=[
                StepSpec(step_id="llm", step_type=StepType.LLM_EXECUTION, step_class="TokenUsingStep"),
            ],
        )

        runtime = RuntimeContext.create()
        result = executor.execute(spec, runtime, initial_data="test")

        assert result.success
        assert result.total_token_usage["prompt_tokens"] == 100
        assert result.total_token_usage["completion_tokens"] == 50
        assert runtime.token_budget.used_prompt_tokens == 100

    def test_token_budget_exceeded_skips_steps(self, executor):
        """Steps are skipped when token budget exceeded."""
        spec = PipelineSpec(
            name="budget",
            version="1.0.0",
            steps=[
                StepSpec(step_id="llm", step_type=StepType.LLM_EXECUTION, step_class="TokenUsingStep"),
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            ],
        )

        runtime = RuntimeContext.create(max_tokens=100)  # Budget of 100, but step uses 150
        result = executor.execute(spec, runtime, initial_data="test")

        # First step executes (goes over budget), second is skipped
        assert len(result.step_results) == 2
        assert result.step_results[0].success
        assert result.step_results[1].skipped
        assert "Budget" in result.step_results[1].skip_reason

    def test_timing_recorded(self, executor):
        """Step timing is recorded."""
        spec = PipelineSpec(
            name="timing",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="slow",
                    step_type=StepType.POST_PROCESSING,
                    step_class="SlowStep",
                    config={"delay_ms": 20},
                ),
            ],
        )

        runtime = RuntimeContext.create()
        result = executor.execute(spec, runtime, initial_data=5)

        assert result.success
        assert result.step_results[0].latency_ms >= 20
        assert runtime.get_step_timing("slow") >= 20
        assert result.total_latency_ms >= 20

    def test_fallback_on_failure(self, executor):
        """Fallback step is executed on failure."""
        spec = PipelineSpec(
            name="fallback",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="fail",
                    step_type=StepType.POST_PROCESSING,
                    step_class="FailingStep",
                    fallback_step="fallback",
                ),
                StepSpec(
                    step_id="fallback",
                    step_type=StepType.POST_PROCESSING,
                    step_class="FallbackStep",
                    enabled=False,  # Only used as fallback
                ),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert result.success
        assert result.output == -5  # Fallback negates
        assert result.step_results[0].metadata.get("fallback_for") == "fail"

    def test_fallback_on_exception(self, executor):
        """Fallback step is executed on exception."""
        spec = PipelineSpec(
            name="fallback-exc",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="exc",
                    step_type=StepType.POST_PROCESSING,
                    step_class="ExceptionStep",
                    fallback_step="fallback",
                ),
                StepSpec(
                    step_id="fallback",
                    step_type=StepType.POST_PROCESSING,
                    step_class="FallbackStep",
                    enabled=False,
                ),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert result.success
        assert result.output == -5

    def test_disabled_steps_skipped(self, executor):
        """Disabled steps are not executed."""
        spec = PipelineSpec(
            name="disabled",
            version="1.0.0",
            steps=[
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
                StepSpec(
                    step_id="disabled",
                    step_type=StepType.POST_PROCESSING,
                    step_class="MultiplyStep",
                    enabled=False,
                ),
                StepSpec(
                    step_id="multiply",
                    step_type=StepType.POST_PROCESSING,
                    step_class="MultiplyStep",
                    config={"factor": 2},
                ),
            ],
        )

        result = executor.execute(spec, initial_data=5)

        assert result.success
        assert result.output == 12  # (5 + 1) * 2, skipping disabled step
        assert len(result.step_results) == 2  # Only enabled steps

    def test_trace_id_propagated(self, executor):
        """Trace ID is propagated through execution."""
        spec = PipelineSpec(
            name="trace",
            version="1.0.0",
            steps=[
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            ],
        )

        runtime = RuntimeContext.create()
        result = executor.execute(spec, runtime, initial_data=5)

        assert result.trace_id == runtime.trace_id
        assert len(result.trace_id) == 36


class TestExecutePipelineFunction:
    """Tests for execute_pipeline convenience function."""

    def test_convenience_function(self, registry):
        """execute_pipeline function works."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            ],
        )

        result = execute_pipeline(spec, initial_data=10, registry=registry)

        assert result.success
        assert result.output == 11
