"""Async and batch execution for RAG pipelines."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from rag_os.core.types import MetadataType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.spec import PipelineSpec, StepSpec
from rag_os.core.registry import StepRegistry, get_registry
from rag_os.core.runtime import RuntimeContext, ExecutionMode, FailureMode
from rag_os.core.executor import (
    PipelineExecutor,
    PipelineResult,
    StepExecutionResult,
)


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class BatchResult:
    """Result of batch pipeline execution."""

    total_items: int
    successful_items: int
    failed_items: int
    results: list[PipelineResult]
    total_latency_ms: float
    started_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None
    metadata: MetadataType = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items

    def get_failed_results(self) -> list[PipelineResult]:
        """Get only failed results."""
        return [r for r in self.results if not r.success]


class AsyncPipelineExecutor:
    """
    Async executor for running RAG pipelines.

    Supports async execution and batch processing of multiple inputs.
    """

    def __init__(
        self,
        registry: StepRegistry | None = None,
        max_concurrency: int = 10,
    ) -> None:
        """
        Initialize the async executor.

        Args:
            registry: Step registry to use
            max_concurrency: Maximum concurrent executions for batch
        """
        self.registry = registry or get_registry()
        self.max_concurrency = max_concurrency
        self._sync_executor = PipelineExecutor(registry)

    async def execute(
        self,
        spec: PipelineSpec,
        runtime: RuntimeContext | None = None,
        initial_data: Any = None,
    ) -> PipelineResult:
        """
        Execute a pipeline asynchronously.

        This wraps the sync executor in an async context,
        running it in a thread pool to avoid blocking.

        Args:
            spec: The pipeline specification
            runtime: Runtime context
            initial_data: Initial data for the pipeline

        Returns:
            PipelineResult with execution results
        """
        if runtime is None:
            runtime = RuntimeContext.create(
                pipeline_name=spec.name,
                pipeline_version=spec.version,
                execution_mode=ExecutionMode.ASYNC,
            )
        else:
            runtime.execution_mode = ExecutionMode.ASYNC

        # Run sync executor in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: self._sync_executor.execute(spec, runtime, initial_data),
            )

        return result

    async def execute_batch(
        self,
        spec: PipelineSpec,
        inputs: list[Any],
        runtime: RuntimeContext | None = None,
    ) -> BatchResult:
        """
        Execute a pipeline for multiple inputs concurrently.

        Args:
            spec: The pipeline specification
            inputs: List of initial data items to process
            runtime: Base runtime context (will be cloned for each execution)

        Returns:
            BatchResult with all execution results
        """
        started_at = _utc_now()
        start_time = time.perf_counter()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_item(item: Any, index: int) -> PipelineResult:
            async with semaphore:
                # Create a new runtime for each item
                item_runtime = RuntimeContext.create(
                    pipeline_name=spec.name,
                    pipeline_version=spec.version,
                    execution_mode=ExecutionMode.BATCH,
                )
                if runtime:
                    item_runtime.user_metadata = runtime.user_metadata.copy()
                    item_runtime.runtime_overrides = runtime.runtime_overrides.copy()
                    item_runtime.failure_mode = runtime.failure_mode

                item_runtime.user_metadata["batch_index"] = index

                return await self.execute(spec, item_runtime, item)

        # Execute all items concurrently
        tasks = [process_item(item, i) for i, item in enumerate(inputs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        pipeline_results: list[PipelineResult] = []
        successful = 0
        failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed PipelineResult
                pipeline_results.append(
                    PipelineResult(
                        success=False,
                        error=str(result),
                        pipeline_name=spec.name,
                        pipeline_version=spec.version,
                        metadata={"batch_index": i, "exception": type(result).__name__},
                    )
                )
                failed += 1
            elif isinstance(result, PipelineResult):
                result.metadata["batch_index"] = i
                pipeline_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1

        total_latency = (time.perf_counter() - start_time) * 1000

        return BatchResult(
            total_items=len(inputs),
            successful_items=successful,
            failed_items=failed,
            results=pipeline_results,
            total_latency_ms=total_latency,
            started_at=started_at,
            completed_at=_utc_now(),
        )

    async def execute_stream(
        self,
        spec: PipelineSpec,
        inputs: list[Any],
        runtime: RuntimeContext | None = None,
    ) -> AsyncIterator[tuple[int, PipelineResult]]:
        """
        Execute a pipeline for multiple inputs, yielding results as they complete.

        Args:
            spec: The pipeline specification
            inputs: List of initial data items to process
            runtime: Base runtime context

        Yields:
            Tuple of (index, result) as each item completes
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_item(item: Any, index: int) -> tuple[int, PipelineResult]:
            async with semaphore:
                item_runtime = RuntimeContext.create(
                    pipeline_name=spec.name,
                    pipeline_version=spec.version,
                    execution_mode=ExecutionMode.BATCH,
                )
                if runtime:
                    item_runtime.user_metadata = runtime.user_metadata.copy()
                    item_runtime.runtime_overrides = runtime.runtime_overrides.copy()

                item_runtime.user_metadata["batch_index"] = index

                try:
                    result = await self.execute(spec, item_runtime, item)
                    return (index, result)
                except Exception as e:
                    return (
                        index,
                        PipelineResult(
                            success=False,
                            error=str(e),
                            pipeline_name=spec.name,
                            pipeline_version=spec.version,
                        ),
                    )

        # Create tasks
        tasks = [
            asyncio.create_task(process_item(item, i))
            for i, item in enumerate(inputs)
        ]

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            index, result = await coro
            yield (index, result)


class DryRunExecutor:
    """
    Executor that simulates pipeline execution without running actual steps.

    Useful for testing, validation, and cost estimation.
    """

    def __init__(self, registry: StepRegistry | None = None) -> None:
        self.registry = registry or get_registry()

    def execute(
        self,
        spec: PipelineSpec,
        runtime: RuntimeContext | None = None,
        initial_data: Any = None,
    ) -> PipelineResult:
        """
        Execute a dry run of the pipeline.

        Args:
            spec: The pipeline specification
            runtime: Runtime context
            initial_data: Initial data

        Returns:
            PipelineResult with simulated execution
        """
        if runtime is None:
            runtime = RuntimeContext.create(
                pipeline_name=spec.name,
                pipeline_version=spec.version,
                execution_mode=ExecutionMode.DRY_RUN,
            )

        result = PipelineResult(
            success=True,
            trace_id=runtime.trace_id,
            pipeline_name=spec.name,
            pipeline_version=spec.version,
            started_at=runtime.start_time,
        )

        data = initial_data
        enabled_steps = spec.get_enabled_steps()

        for step_spec in enabled_steps:
            # Check if step exists
            step_class = self.registry.get(step_spec.step_class)
            if step_class is None:
                result.step_results.append(
                    StepExecutionResult(
                        step_id=step_spec.step_id,
                        step_type=str(step_spec.step_type),
                        success=False,
                        error=f"Step class '{step_spec.step_class}' not found",
                    )
                )
                result.success = False
                continue

            # Simulate successful execution
            result.step_results.append(
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=True,
                    output=data,  # Pass through data unchanged
                    latency_ms=0.0,
                    metadata={"dry_run": True},
                )
            )

        result.completed_at = _utc_now()
        result.total_latency_ms = runtime.elapsed_ms
        result.output = data
        result.metadata["dry_run"] = True

        return result

    def estimate_cost(
        self,
        spec: PipelineSpec,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict[str, float]:
        """
        Estimate the cost of running a pipeline.

        This is a simplified estimation based on step types.
        Actual implementation would need pricing data.

        Args:
            spec: The pipeline specification
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Dict with cost estimates by category
        """
        # Placeholder cost estimation
        # In production, this would use actual pricing data
        estimates = {
            "embedding_cost": 0.0,
            "llm_cost": 0.0,
            "reranking_cost": 0.0,
            "total_cost": 0.0,
        }

        from rag_os.core.types import StepType

        for step in spec.get_enabled_steps():
            if step.step_type == StepType.EMBEDDING:
                # $0.00002 per 1K tokens (example rate)
                estimates["embedding_cost"] += (input_tokens / 1000) * 0.00002
            elif step.step_type == StepType.LLM_EXECUTION:
                # $0.01 per 1K input, $0.03 per 1K output (example rates)
                estimates["llm_cost"] += (input_tokens / 1000) * 0.01
                estimates["llm_cost"] += (output_tokens / 1000) * 0.03
            elif step.step_type == StepType.RERANKING:
                # $0.0001 per query (example rate)
                estimates["reranking_cost"] += 0.0001

        estimates["total_cost"] = sum(estimates.values())
        return estimates


async def execute_pipeline_async(
    spec: PipelineSpec,
    initial_data: Any = None,
    runtime: RuntimeContext | None = None,
    registry: StepRegistry | None = None,
) -> PipelineResult:
    """
    Convenience function for async pipeline execution.

    Args:
        spec: Pipeline specification
        initial_data: Initial data
        runtime: Runtime context
        registry: Step registry

    Returns:
        PipelineResult
    """
    executor = AsyncPipelineExecutor(registry)
    return await executor.execute(spec, runtime, initial_data)


async def execute_batch(
    spec: PipelineSpec,
    inputs: list[Any],
    runtime: RuntimeContext | None = None,
    registry: StepRegistry | None = None,
    max_concurrency: int = 10,
) -> BatchResult:
    """
    Convenience function for batch pipeline execution.

    Args:
        spec: Pipeline specification
        inputs: List of inputs to process
        runtime: Runtime context
        registry: Step registry
        max_concurrency: Maximum concurrent executions

    Returns:
        BatchResult
    """
    executor = AsyncPipelineExecutor(registry, max_concurrency)
    return await executor.execute_batch(spec, inputs, runtime)
