"""Runtime context for pipeline execution."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from enum import Enum

from rag_os.core.types import MetadataType


class ExecutionMode(str, Enum):
    """Mode of pipeline execution."""

    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    DRY_RUN = "dry_run"


class FailureMode(str, Enum):
    """How to handle step failures."""

    FAIL_FAST = "fail_fast"  # Stop immediately on first failure
    GRACEFUL = "graceful"  # Continue with remaining steps, skip failed branches


@dataclass
class TokenBudget:
    """Token budget constraints for pipeline execution."""

    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None

    # Track actual usage
    used_prompt_tokens: int = 0
    used_completion_tokens: int = 0

    @property
    def used_total_tokens(self) -> int:
        return self.used_prompt_tokens + self.used_completion_tokens

    @property
    def remaining_prompt_tokens(self) -> int | None:
        if self.max_prompt_tokens is None:
            return None
        return max(0, self.max_prompt_tokens - self.used_prompt_tokens)

    @property
    def remaining_completion_tokens(self) -> int | None:
        if self.max_completion_tokens is None:
            return None
        return max(0, self.max_completion_tokens - self.used_completion_tokens)

    @property
    def remaining_total_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return max(0, self.max_total_tokens - self.used_total_tokens)

    def is_exceeded(self) -> bool:
        """Check if any budget limit is exceeded."""
        if self.max_prompt_tokens and self.used_prompt_tokens > self.max_prompt_tokens:
            return True
        if self.max_completion_tokens and self.used_completion_tokens > self.max_completion_tokens:
            return True
        if self.max_total_tokens and self.used_total_tokens > self.max_total_tokens:
            return True
        return False

    def add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record token usage."""
        self.used_prompt_tokens += prompt_tokens
        self.used_completion_tokens += completion_tokens


@dataclass
class CostBudget:
    """Cost budget constraints for pipeline execution."""

    max_cost_usd: float | None = None
    used_cost_usd: float = 0.0

    @property
    def remaining_cost_usd(self) -> float | None:
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.used_cost_usd)

    def is_exceeded(self) -> bool:
        """Check if cost budget is exceeded."""
        if self.max_cost_usd is None:
            return False
        return self.used_cost_usd > self.max_cost_usd

    def add_cost(self, cost_usd: float) -> None:
        """Record cost."""
        self.used_cost_usd += cost_usd


@dataclass
class RBACScope:
    """Role-based access control scope for the execution."""

    user_id: str | None = None
    roles: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class RuntimeContext:
    """
    Runtime context for pipeline execution.

    Contains all runtime information needed during pipeline execution,
    including user metadata, budgets, RBAC scope, and execution settings.

    This is separate from StepContext - RuntimeContext is for the entire
    pipeline execution, while StepContext carries data between steps.

    Attributes:
        trace_id: Unique identifier for this execution (for tracing/logging)
        pipeline_name: Name of the pipeline being executed
        pipeline_version: Version of the pipeline being executed
        execution_mode: How the pipeline is being executed
        failure_mode: How to handle step failures
        user_metadata: User-provided metadata
        rbac_scope: Role-based access control scope
        token_budget: Token usage constraints
        cost_budget: Cost constraints
        runtime_overrides: Config overrides to apply at runtime
        start_time: When execution started
        tags: Tags for categorization/filtering
    """

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    pipeline_name: str = ""
    pipeline_version: str = ""
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    failure_mode: FailureMode = FailureMode.FAIL_FAST
    user_metadata: MetadataType = field(default_factory=dict)
    rbac_scope: RBACScope = field(default_factory=RBACScope)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    cost_budget: CostBudget = field(default_factory=CostBudget)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=_utc_now)
    tags: list[str] = field(default_factory=list)

    # Internal state
    _current_step_id: str | None = field(default=None, repr=False)
    _step_timings: dict[str, float] = field(default_factory=dict, repr=False)

    def get_step_override(self, step_id: str, key: str, default: Any = None) -> Any:
        """Get a runtime override for a specific step."""
        step_overrides = self.runtime_overrides.get(step_id, {})
        return step_overrides.get(key, default)

    def get_global_override(self, key: str, default: Any = None) -> Any:
        """Get a global runtime override."""
        return self.runtime_overrides.get(key, default)

    def set_current_step(self, step_id: str) -> None:
        """Set the currently executing step (for logging/tracing)."""
        self._current_step_id = step_id

    def record_step_timing(self, step_id: str, latency_ms: float) -> None:
        """Record timing for a step."""
        self._step_timings[step_id] = latency_ms

    def get_step_timing(self, step_id: str) -> float | None:
        """Get recorded timing for a step."""
        return self._step_timings.get(step_id)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        delta = _utc_now() - self.start_time
        return delta.total_seconds() * 1000

    @property
    def total_step_time_ms(self) -> float:
        """Get total time spent in steps."""
        return sum(self._step_timings.values())

    def is_budget_exceeded(self) -> bool:
        """Check if any budget (token or cost) is exceeded."""
        return self.token_budget.is_exceeded() or self.cost_budget.is_exceeded()

    def is_dry_run(self) -> bool:
        """Check if this is a dry run execution."""
        return self.execution_mode == ExecutionMode.DRY_RUN

    @classmethod
    def create(
        cls,
        pipeline_name: str = "",
        pipeline_version: str = "",
        user_id: str | None = None,
        execution_mode: ExecutionMode | str = ExecutionMode.SYNC,
        failure_mode: FailureMode | str = FailureMode.FAIL_FAST,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        overrides: dict[str, Any] | None = None,
        **user_metadata: Any,
    ) -> "RuntimeContext":
        """
        Factory method to create a RuntimeContext with common settings.

        Args:
            pipeline_name: Name of the pipeline
            pipeline_version: Version of the pipeline
            user_id: User ID for RBAC
            execution_mode: Execution mode (sync, async, batch, dry_run)
            failure_mode: Failure handling mode
            max_tokens: Maximum total tokens allowed
            max_cost_usd: Maximum cost in USD allowed
            overrides: Runtime config overrides
            **user_metadata: Additional user metadata

        Returns:
            Configured RuntimeContext
        """
        if isinstance(execution_mode, str):
            execution_mode = ExecutionMode(execution_mode)
        if isinstance(failure_mode, str):
            failure_mode = FailureMode(failure_mode)

        return cls(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            execution_mode=execution_mode,
            failure_mode=failure_mode,
            user_metadata=user_metadata,
            rbac_scope=RBACScope(user_id=user_id) if user_id else RBACScope(),
            token_budget=TokenBudget(max_total_tokens=max_tokens) if max_tokens else TokenBudget(),
            cost_budget=CostBudget(max_cost_usd=max_cost_usd) if max_cost_usd else CostBudget(),
            runtime_overrides=overrides or {},
        )
