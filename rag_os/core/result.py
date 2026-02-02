"""Step result container for pipeline step outputs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rag_os.core.types import MetadataType


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class StepResult:
    """
    Result object returned by each pipeline step after execution.

    Attributes:
        output: The primary output data from the step
        success: Whether the step executed successfully
        error: Error message if the step failed
        metadata: Additional metadata about the execution
        latency_ms: Execution time in milliseconds
        token_usage: Token usage if applicable (for LLM steps)
        created_at: Timestamp when this result was created
    """

    output: Any = None
    success: bool = True
    error: str | None = None
    metadata: MetadataType = field(default_factory=dict)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    @classmethod
    def ok(cls, output: Any, **metadata: Any) -> "StepResult":
        """Create a successful result with the given output."""
        return cls(output=output, success=True, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "StepResult":
        """Create a failed result with the given error message."""
        return cls(output=None, success=False, error=error, metadata=metadata)

    def with_latency(self, latency_ms: float) -> "StepResult":
        """Return a new result with the specified latency."""
        return StepResult(
            output=self.output,
            success=self.success,
            error=self.error,
            metadata=self.metadata.copy(),
            latency_ms=latency_ms,
            token_usage=self.token_usage.copy(),
            created_at=self.created_at,
        )

    def with_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> "StepResult":
        """Return a new result with token usage information."""
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        }
        return StepResult(
            output=self.output,
            success=self.success,
            error=self.error,
            metadata=self.metadata.copy(),
            latency_ms=self.latency_ms,
            token_usage=usage,
            created_at=self.created_at,
        )
