"""Unit tests for runtime context."""

import pytest
from datetime import datetime, timezone
import time

from rag_os.core.runtime import (
    RuntimeContext,
    ExecutionMode,
    FailureMode,
    TokenBudget,
    CostBudget,
    RBACScope,
)


class TestTokenBudget:
    """Tests for TokenBudget."""

    def test_default_no_limits(self):
        """Default budget has no limits."""
        budget = TokenBudget()
        assert budget.max_prompt_tokens is None
        assert budget.max_total_tokens is None
        assert not budget.is_exceeded()

    def test_track_usage(self):
        """Budget tracks token usage."""
        budget = TokenBudget()
        budget.add_usage(prompt_tokens=100, completion_tokens=50)

        assert budget.used_prompt_tokens == 100
        assert budget.used_completion_tokens == 50
        assert budget.used_total_tokens == 150

    def test_remaining_tokens(self):
        """Remaining tokens are calculated correctly."""
        budget = TokenBudget(max_total_tokens=1000)
        budget.add_usage(prompt_tokens=300, completion_tokens=200)

        assert budget.remaining_total_tokens == 500

    def test_exceeded_prompt_tokens(self):
        """Exceeding prompt tokens is detected."""
        budget = TokenBudget(max_prompt_tokens=100)
        budget.add_usage(prompt_tokens=150)

        assert budget.is_exceeded()

    def test_exceeded_total_tokens(self):
        """Exceeding total tokens is detected."""
        budget = TokenBudget(max_total_tokens=200)
        budget.add_usage(prompt_tokens=100, completion_tokens=150)

        assert budget.is_exceeded()

    def test_not_exceeded_within_limits(self):
        """Within limits is not exceeded."""
        budget = TokenBudget(max_total_tokens=1000)
        budget.add_usage(prompt_tokens=400, completion_tokens=300)

        assert not budget.is_exceeded()


class TestCostBudget:
    """Tests for CostBudget."""

    def test_default_no_limit(self):
        """Default budget has no limit."""
        budget = CostBudget()
        assert budget.max_cost_usd is None
        assert not budget.is_exceeded()

    def test_track_cost(self):
        """Budget tracks cost."""
        budget = CostBudget(max_cost_usd=10.0)
        budget.add_cost(2.50)
        budget.add_cost(1.25)

        assert budget.used_cost_usd == 3.75
        assert budget.remaining_cost_usd == 6.25

    def test_exceeded(self):
        """Exceeding cost is detected."""
        budget = CostBudget(max_cost_usd=5.0)
        budget.add_cost(6.0)

        assert budget.is_exceeded()


class TestRBACScope:
    """Tests for RBACScope."""

    def test_default_empty(self):
        """Default scope is empty."""
        scope = RBACScope()
        assert scope.user_id is None
        assert scope.roles == []
        assert scope.permissions == []

    def test_has_role(self):
        """Role checking works."""
        scope = RBACScope(roles=["admin", "user"])

        assert scope.has_role("admin")
        assert scope.has_role("user")
        assert not scope.has_role("superuser")

    def test_has_permission(self):
        """Permission checking works."""
        scope = RBACScope(permissions=["read", "write"])

        assert scope.has_permission("read")
        assert not scope.has_permission("delete")


class TestRuntimeContext:
    """Tests for RuntimeContext."""

    def test_default_creation(self):
        """Context can be created with defaults."""
        ctx = RuntimeContext()

        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 36  # UUID format
        assert ctx.execution_mode == ExecutionMode.SYNC
        assert ctx.failure_mode == FailureMode.FAIL_FAST
        assert ctx.user_metadata == {}

    def test_factory_method(self):
        """Factory method creates context with common settings."""
        ctx = RuntimeContext.create(
            pipeline_name="test-pipeline",
            pipeline_version="1.0.0",
            user_id="user123",
            max_tokens=10000,
            max_cost_usd=5.0,
            session_id="session456",
        )

        assert ctx.pipeline_name == "test-pipeline"
        assert ctx.pipeline_version == "1.0.0"
        assert ctx.rbac_scope.user_id == "user123"
        assert ctx.token_budget.max_total_tokens == 10000
        assert ctx.cost_budget.max_cost_usd == 5.0
        assert ctx.user_metadata["session_id"] == "session456"

    def test_execution_mode_string(self):
        """Execution mode can be provided as string."""
        ctx = RuntimeContext.create(execution_mode="async")
        assert ctx.execution_mode == ExecutionMode.ASYNC

    def test_failure_mode_string(self):
        """Failure mode can be provided as string."""
        ctx = RuntimeContext.create(failure_mode="graceful")
        assert ctx.failure_mode == FailureMode.GRACEFUL

    def test_runtime_overrides(self):
        """Runtime overrides work."""
        ctx = RuntimeContext.create(
            overrides={
                "step1": {"temperature": 0.5},
                "step2": {"top_k": 10},
                "global_setting": "value",
            }
        )

        assert ctx.get_step_override("step1", "temperature") == 0.5
        assert ctx.get_step_override("step1", "missing", "default") == "default"
        assert ctx.get_step_override("step2", "top_k") == 10
        assert ctx.get_global_override("global_setting") == "value"

    def test_step_timing(self):
        """Step timing recording works."""
        ctx = RuntimeContext()
        ctx.record_step_timing("step1", 100.5)
        ctx.record_step_timing("step2", 200.0)

        assert ctx.get_step_timing("step1") == 100.5
        assert ctx.get_step_timing("step2") == 200.0
        assert ctx.get_step_timing("nonexistent") is None
        assert ctx.total_step_time_ms == 300.5

    def test_current_step(self):
        """Current step tracking works."""
        ctx = RuntimeContext()
        assert ctx._current_step_id is None

        ctx.set_current_step("step1")
        assert ctx._current_step_id == "step1"

    def test_elapsed_time(self):
        """Elapsed time is calculated."""
        ctx = RuntimeContext()
        time.sleep(0.01)  # 10ms

        assert ctx.elapsed_ms >= 10

    def test_budget_exceeded_tokens(self):
        """Budget exceeded check includes tokens."""
        ctx = RuntimeContext.create(max_tokens=100)
        ctx.token_budget.add_usage(prompt_tokens=150)

        assert ctx.is_budget_exceeded()

    def test_budget_exceeded_cost(self):
        """Budget exceeded check includes cost."""
        ctx = RuntimeContext.create(max_cost_usd=1.0)
        ctx.cost_budget.add_cost(2.0)

        assert ctx.is_budget_exceeded()

    def test_budget_not_exceeded(self):
        """Budget not exceeded when within limits."""
        ctx = RuntimeContext.create(max_tokens=1000, max_cost_usd=10.0)
        ctx.token_budget.add_usage(prompt_tokens=100)
        ctx.cost_budget.add_cost(1.0)

        assert not ctx.is_budget_exceeded()

    def test_is_dry_run(self):
        """Dry run detection works."""
        ctx_normal = RuntimeContext.create(execution_mode="sync")
        ctx_dry = RuntimeContext.create(execution_mode="dry_run")

        assert not ctx_normal.is_dry_run()
        assert ctx_dry.is_dry_run()

    def test_start_time_is_set(self):
        """Start time is set on creation."""
        before = datetime.now(timezone.utc)
        ctx = RuntimeContext()
        after = datetime.now(timezone.utc)

        assert before <= ctx.start_time <= after

    def test_tags(self):
        """Tags can be set."""
        ctx = RuntimeContext(tags=["production", "high-priority"])

        assert "production" in ctx.tags
        assert "high-priority" in ctx.tags


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_all_modes(self):
        """All execution modes exist."""
        assert ExecutionMode.SYNC.value == "sync"
        assert ExecutionMode.ASYNC.value == "async"
        assert ExecutionMode.BATCH.value == "batch"
        assert ExecutionMode.DRY_RUN.value == "dry_run"


class TestFailureMode:
    """Tests for FailureMode enum."""

    def test_all_modes(self):
        """All failure modes exist."""
        assert FailureMode.FAIL_FAST.value == "fail_fast"
        assert FailureMode.GRACEFUL.value == "graceful"
