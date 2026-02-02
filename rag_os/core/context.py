"""Step context for data propagation between pipeline steps."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

from rag_os.core.types import MetadataType


@dataclass
class StepContext:
    """
    Context object that carries data between pipeline steps.

    This is the primary mechanism for passing data through the pipeline.
    Each step receives a context, processes it, and the result is used
    to create the context for the next step.

    Attributes:
        data: The primary data payload being processed
        metadata: Additional metadata about the current execution
        trace_id: Unique identifier for tracing this execution
        user_metadata: User-provided metadata (e.g., user_id, session_id)
        pipeline_version: Version of the pipeline being executed
        step_outputs: Accumulated outputs from previous steps (keyed by step_id)
        created_at: Timestamp when this context was created
    """

    data: Any = None
    metadata: MetadataType = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    user_metadata: MetadataType = field(default_factory=dict)
    pipeline_version: str = ""
    step_outputs: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def with_data(self, data: Any) -> "StepContext":
        """Create a new context with updated data, preserving other fields."""
        return StepContext(
            data=data,
            metadata=self.metadata.copy(),
            trace_id=self.trace_id,
            user_metadata=self.user_metadata.copy(),
            pipeline_version=self.pipeline_version,
            step_outputs=self.step_outputs.copy(),
            created_at=self.created_at,
        )

    def with_step_output(self, step_id: str, output: Any) -> "StepContext":
        """Create a new context with an additional step output recorded."""
        new_outputs = self.step_outputs.copy()
        new_outputs[step_id] = output
        return StepContext(
            data=self.data,
            metadata=self.metadata.copy(),
            trace_id=self.trace_id,
            user_metadata=self.user_metadata.copy(),
            pipeline_version=self.pipeline_version,
            step_outputs=new_outputs,
            created_at=self.created_at,
        )

    def get_step_output(self, step_id: str) -> Any | None:
        """Retrieve the output of a previous step by its ID."""
        return self.step_outputs.get(step_id)

    def update_metadata(self, **kwargs: Any) -> "StepContext":
        """Create a new context with updated metadata."""
        new_metadata = self.metadata.copy()
        new_metadata.update(kwargs)
        return StepContext(
            data=self.data,
            metadata=new_metadata,
            trace_id=self.trace_id,
            user_metadata=self.user_metadata.copy(),
            pipeline_version=self.pipeline_version,
            step_outputs=self.step_outputs.copy(),
            created_at=self.created_at,
        )
