"""Pipeline executor for running RAG pipelines."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rag_os.core.types import MetadataType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.spec import PipelineSpec, StepSpec
from rag_os.core.registry import StepRegistry, get_registry
from rag_os.core.runtime import RuntimeContext, ExecutionMode, FailureMode


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class StepExecutionResult:
    """Result of executing a single step."""

    step_id: str
    step_type: str
    success: bool
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: MetadataType = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class PipelineResult:
    """Result of executing a complete pipeline."""

    success: bool
    output: Any = None
    error: str | None = None
    trace_id: str = ""
    pipeline_name: str = ""
    pipeline_version: str = ""
    step_results: list[StepExecutionResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_token_usage: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None
    metadata: MetadataType = field(default_factory=dict)

    def get_step_result(self, step_id: str) -> StepExecutionResult | None:
        """Get result for a specific step."""
        for result in self.step_results:
            if result.step_id == step_id:
                return result
        return None

    @property
    def failed_steps(self) -> list[StepExecutionResult]:
        """Get all failed step results."""
        return [r for r in self.step_results if not r.success and not r.skipped]

    @property
    def skipped_steps(self) -> list[StepExecutionResult]:
        """Get all skipped step results."""
        return [r for r in self.step_results if r.skipped]

    @property
    def successful_steps(self) -> list[StepExecutionResult]:
        """Get all successful step results."""
        return [r for r in self.step_results if r.success and not r.skipped]


class PipelineExecutor:
    """
    Executor for running RAG pipelines synchronously.

    The executor takes a pipeline specification and executes each step
    in order, passing context between steps and collecting results.

    Example:
        executor = PipelineExecutor()
        result = executor.execute(pipeline_spec, runtime_context, initial_data)
    """

    def __init__(self, registry: StepRegistry | None = None) -> None:
        """
        Initialize the executor.

        Args:
            registry: Step registry to use. If None, uses the global registry.
        """
        self.registry = registry or get_registry()

    def execute(
        self,
        spec: PipelineSpec,
        runtime: RuntimeContext | None = None,
        initial_data: Any = None,
    ) -> PipelineResult:
        """
        Execute a pipeline synchronously.

        Args:
            spec: The pipeline specification to execute
            runtime: Runtime context (created if not provided)
            initial_data: Initial data to pass to the first step

        Returns:
            PipelineResult with execution results
        """
        # Create runtime context if not provided
        if runtime is None:
            runtime = RuntimeContext.create(
                pipeline_name=spec.name,
                pipeline_version=spec.version,
            )
        else:
            runtime.pipeline_name = spec.name
            runtime.pipeline_version = spec.version

        # Initialize pipeline result
        result = PipelineResult(
            success=True,
            trace_id=runtime.trace_id,
            pipeline_name=spec.name,
            pipeline_version=spec.version,
            started_at=runtime.start_time,
        )

        # Initialize step context with initial data
        context = StepContext(
            data=initial_data,
            trace_id=runtime.trace_id,
            user_metadata=runtime.user_metadata,
            pipeline_version=spec.version,
        )

        # Get enabled steps
        enabled_steps = spec.get_enabled_steps()

        # Execute each step
        for step_spec in enabled_steps:
            # Check if budget exceeded
            if runtime.is_budget_exceeded():
                step_result = StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    skipped=True,
                    skip_reason="Budget exceeded",
                )
                result.step_results.append(step_result)
                continue

            # Execute the step
            step_result, context = self._execute_step(
                step_spec=step_spec,
                context=context,
                runtime=runtime,
                spec=spec,
            )

            result.step_results.append(step_result)

            # Update token usage totals
            for key, value in step_result.token_usage.items():
                result.total_token_usage[key] = result.total_token_usage.get(key, 0) + value

            # Handle step failure
            if not step_result.success and not step_result.skipped:
                if runtime.failure_mode == FailureMode.FAIL_FAST:
                    result.success = False
                    result.error = f"Step '{step_spec.step_id}' failed: {step_result.error}"
                    break
                # In graceful mode, continue execution

        # Finalize result
        result.completed_at = _utc_now()
        result.total_latency_ms = runtime.elapsed_ms
        result.output = context.data

        # Set final success status
        if result.success and result.failed_steps:
            result.success = False
            result.error = f"{len(result.failed_steps)} step(s) failed"

        return result

    def _execute_step(
        self,
        step_spec: StepSpec,
        context: StepContext,
        runtime: RuntimeContext,
        spec: PipelineSpec,
    ) -> tuple[StepExecutionResult, StepContext]:
        """
        Execute a single step.

        Returns:
            Tuple of (step execution result, updated context)
        """
        runtime.set_current_step(step_spec.step_id)

        # Get the step class from registry
        step_class = self.registry.get(step_spec.step_class)
        if step_class is None:
            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=f"Step class '{step_spec.step_class}' not found in registry",
                ),
                context,
            )

        # Merge config with runtime overrides
        config = step_spec.config.copy()
        step_overrides = runtime.runtime_overrides.get(step_spec.step_id, {})
        config.update(step_overrides)

        # Instantiate the step
        try:
            step = step_class(
                step_id=step_spec.step_id,
                step_type=step_spec.step_type,
                config=config,
            )
        except Exception as e:
            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=f"Failed to instantiate step: {e}",
                ),
                context,
            )

        # Validate input
        input_errors = step.validate_input(context)
        if input_errors:
            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=f"Input validation failed: {'; '.join(input_errors)}",
                ),
                context,
            )

        # Execute the step (with timing)
        start_time = time.perf_counter()
        try:
            if runtime.is_dry_run():
                # In dry run mode, skip actual execution
                step_result = StepResult.ok(
                    context.data,
                    dry_run=True,
                    step_id=step_spec.step_id,
                )
            else:
                step_result = step.execute(context)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            runtime.record_step_timing(step_spec.step_id, latency_ms)

            # Try fallback if configured
            if step_spec.fallback_step:
                return self._execute_fallback(
                    step_spec=step_spec,
                    context=context,
                    runtime=runtime,
                    spec=spec,
                    original_error=str(e),
                )

            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                ),
                context,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        runtime.record_step_timing(step_spec.step_id, latency_ms)

        # Handle step failure
        if not step_result.success:
            # Try fallback if configured
            if step_spec.fallback_step:
                return self._execute_fallback(
                    step_spec=step_spec,
                    context=context,
                    runtime=runtime,
                    spec=spec,
                    original_error=step_result.error or "Unknown error",
                )

            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=step_result.error,
                    latency_ms=latency_ms,
                ),
                context,
            )

        # Validate output
        output_errors = step.validate_output(step_result)
        if output_errors:
            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=f"Output validation failed: {'; '.join(output_errors)}",
                    latency_ms=latency_ms,
                ),
                context,
            )

        # Update runtime with token usage
        if step_result.token_usage:
            runtime.token_budget.add_usage(
                prompt_tokens=step_result.token_usage.get("prompt_tokens", 0),
                completion_tokens=step_result.token_usage.get("completion_tokens", 0),
            )

        # Update context with step output
        new_context = context.with_data(step_result.output)
        new_context = new_context.with_step_output(step_spec.step_id, step_result.output)

        return (
            StepExecutionResult(
                step_id=step_spec.step_id,
                step_type=str(step_spec.step_type),
                success=True,
                output=step_result.output,
                latency_ms=latency_ms,
                token_usage=step_result.token_usage,
                metadata=step_result.metadata,
            ),
            new_context,
        )

    def _execute_fallback(
        self,
        step_spec: StepSpec,
        context: StepContext,
        runtime: RuntimeContext,
        spec: PipelineSpec,
        original_error: str,
    ) -> tuple[StepExecutionResult, StepContext]:
        """Execute a fallback step after the primary step fails."""
        fallback_spec = spec.get_step(step_spec.fallback_step)  # type: ignore
        if fallback_spec is None:
            return (
                StepExecutionResult(
                    step_id=step_spec.step_id,
                    step_type=str(step_spec.step_type),
                    success=False,
                    error=f"{original_error} (fallback step not found)",
                ),
                context,
            )

        # Execute fallback
        fallback_result, new_context = self._execute_step(
            step_spec=fallback_spec,
            context=context,
            runtime=runtime,
            spec=spec,
        )

        # Annotate result to indicate fallback was used
        fallback_result.metadata["fallback_for"] = step_spec.step_id
        fallback_result.metadata["original_error"] = original_error

        return fallback_result, new_context


def execute_pipeline(
    spec: PipelineSpec,
    initial_data: Any = None,
    runtime: RuntimeContext | None = None,
    registry: StepRegistry | None = None,
) -> PipelineResult:
    """
    Convenience function to execute a pipeline.

    Args:
        spec: Pipeline specification
        initial_data: Initial data for the pipeline
        runtime: Runtime context (optional)
        registry: Step registry (optional)

    Returns:
        PipelineResult with execution results
    """
    executor = PipelineExecutor(registry)
    return executor.execute(spec, runtime, initial_data)
