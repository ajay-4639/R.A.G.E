"""Unit tests for async and batch execution."""

import pytest
import asyncio

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.spec import StepSpec, PipelineSpec
from rag_os.core.registry import StepRegistry
from rag_os.core.runtime import RuntimeContext, ExecutionMode
from rag_os.core.async_executor import (
    AsyncPipelineExecutor,
    DryRunExecutor,
    BatchResult,
    execute_pipeline_async,
    execute_batch,
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

    def execute(self, context: StepContext) -> StepResult:
        factor = self.config.get("factor", 2)
        return StepResult.ok(context.data * factor)


class FailOnNegativeStep(Step):
    """Fails if input is negative."""

    @property
    def input_schema(self) -> dict:
        return {"type": "number"}

    @property
    def output_schema(self) -> dict:
        return {"type": "number"}

    def execute(self, context: StepContext) -> StepResult:
        if context.data < 0:
            return StepResult.fail("Negative input not allowed")
        return StepResult.ok(context.data)


@pytest.fixture
def registry():
    """Provide a registry with test steps."""
    reg = StepRegistry()
    reg.clear()

    reg.register(AddOneStep, "AddOneStep", StepType.POST_PROCESSING)
    reg.register(MultiplyStep, "MultiplyStep", StepType.POST_PROCESSING)
    reg.register(FailOnNegativeStep, "FailOnNegativeStep", StepType.POST_PROCESSING)

    yield reg
    reg.clear()


@pytest.fixture
def simple_spec():
    """Provide a simple pipeline spec."""
    return PipelineSpec(
        name="simple",
        version="1.0.0",
        steps=[
            StepSpec(step_id="add", step_type=StepType.POST_PROCESSING, step_class="AddOneStep"),
            StepSpec(
                step_id="multiply",
                step_type=StepType.POST_PROCESSING,
                step_class="MultiplyStep",
                config={"factor": 2},
            ),
        ],
    )


class TestBatchResult:
    """Tests for BatchResult."""

    def test_success_rate(self):
        """Success rate is calculated correctly."""
        result = BatchResult(
            total_items=10,
            successful_items=7,
            failed_items=3,
            results=[],
            total_latency_ms=1000.0,
        )

        assert result.success_rate == 0.7

    def test_success_rate_empty(self):
        """Success rate is 0 for empty batch."""
        result = BatchResult(
            total_items=0,
            successful_items=0,
            failed_items=0,
            results=[],
            total_latency_ms=0.0,
        )

        assert result.success_rate == 0.0


class TestAsyncPipelineExecutor:
    """Tests for AsyncPipelineExecutor."""

    @pytest.mark.asyncio
    async def test_async_execute(self, registry, simple_spec):
        """Async execution works."""
        executor = AsyncPipelineExecutor(registry)
        result = await executor.execute(simple_spec, initial_data=5)

        assert result.success
        assert result.output == 12  # (5 + 1) * 2

    @pytest.mark.asyncio
    async def test_async_execution_mode(self, registry, simple_spec):
        """Execution mode is set to async."""
        executor = AsyncPipelineExecutor(registry)
        runtime = RuntimeContext.create()

        await executor.execute(simple_spec, runtime, initial_data=5)

        assert runtime.execution_mode == ExecutionMode.ASYNC

    @pytest.mark.asyncio
    async def test_batch_execute(self, registry, simple_spec):
        """Batch execution processes multiple inputs."""
        executor = AsyncPipelineExecutor(registry)
        inputs = [1, 2, 3, 4, 5]

        result = await executor.execute_batch(simple_spec, inputs)

        assert result.total_items == 5
        assert result.successful_items == 5
        assert result.failed_items == 0
        assert len(result.results) == 5

        # Check outputs: (n + 1) * 2
        outputs = [r.output for r in result.results]
        assert set(outputs) == {4, 6, 8, 10, 12}

    @pytest.mark.asyncio
    async def test_batch_with_failures(self, registry):
        """Batch handles failures in individual items."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="check",
                    step_type=StepType.POST_PROCESSING,
                    step_class="FailOnNegativeStep",
                ),
            ],
        )

        executor = AsyncPipelineExecutor(registry)
        inputs = [1, -2, 3, -4, 5]

        result = await executor.execute_batch(spec, inputs)

        assert result.total_items == 5
        assert result.successful_items == 3
        assert result.failed_items == 2

    @pytest.mark.asyncio
    async def test_batch_concurrency(self, registry, simple_spec):
        """Batch respects max concurrency."""
        executor = AsyncPipelineExecutor(registry, max_concurrency=2)
        inputs = list(range(10))

        result = await executor.execute_batch(simple_spec, inputs)

        assert result.total_items == 10
        assert result.successful_items == 10

    @pytest.mark.asyncio
    async def test_batch_metadata(self, registry, simple_spec):
        """Batch adds index metadata to results."""
        executor = AsyncPipelineExecutor(registry)
        inputs = [1, 2, 3]

        result = await executor.execute_batch(simple_spec, inputs)

        indices = {r.metadata.get("batch_index") for r in result.results}
        assert indices == {0, 1, 2}

    @pytest.mark.asyncio
    async def test_stream_execute(self, registry, simple_spec):
        """Stream execution yields results as they complete."""
        executor = AsyncPipelineExecutor(registry)
        inputs = [1, 2, 3]

        results = []
        async for index, result in executor.execute_stream(simple_spec, inputs):
            results.append((index, result))

        assert len(results) == 3
        indices = {r[0] for r in results}
        assert indices == {0, 1, 2}


class TestDryRunExecutor:
    """Tests for DryRunExecutor."""

    def test_dry_run_succeeds(self, registry, simple_spec):
        """Dry run completes without actual execution."""
        executor = DryRunExecutor(registry)
        result = executor.execute(simple_spec, initial_data=5)

        assert result.success
        assert result.metadata.get("dry_run") is True
        assert len(result.step_results) == 2

    def test_dry_run_detects_missing_steps(self, registry):
        """Dry run detects missing step classes."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="missing",
                    step_type=StepType.POST_PROCESSING,
                    step_class="NonExistentStep",
                ),
            ],
        )

        executor = DryRunExecutor(registry)
        result = executor.execute(spec, initial_data=5)

        assert not result.success
        assert "not found" in result.step_results[0].error

    def test_dry_run_passthrough_data(self, registry, simple_spec):
        """Dry run passes through data unchanged."""
        executor = DryRunExecutor(registry)
        result = executor.execute(simple_spec, initial_data={"test": "data"})

        assert result.output == {"test": "data"}

    def test_estimate_cost(self, registry):
        """Cost estimation returns estimates."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(step_id="embed", step_type=StepType.EMBEDDING, step_class="AddOneStep"),
                StepSpec(step_id="llm", step_type=StepType.LLM_EXECUTION, step_class="AddOneStep"),
            ],
        )

        executor = DryRunExecutor(registry)
        estimates = executor.estimate_cost(spec, input_tokens=1000, output_tokens=500)

        assert "embedding_cost" in estimates
        assert "llm_cost" in estimates
        assert "total_cost" in estimates
        assert estimates["total_cost"] > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_pipeline_async(self, registry, simple_spec):
        """execute_pipeline_async convenience function works."""
        result = await execute_pipeline_async(
            simple_spec,
            initial_data=5,
            registry=registry,
        )

        assert result.success
        assert result.output == 12

    @pytest.mark.asyncio
    async def test_execute_batch(self, registry, simple_spec):
        """execute_batch convenience function works."""
        result = await execute_batch(
            simple_spec,
            inputs=[1, 2, 3],
            registry=registry,
            max_concurrency=2,
        )

        assert result.total_items == 3
        assert result.successful_items == 3
