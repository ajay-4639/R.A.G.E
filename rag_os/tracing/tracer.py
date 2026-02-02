"""Distributed tracing for RAG OS pipelines."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
import threading


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for distributed tracing.

    Attributes:
        trace_id: Unique ID for the entire trace
        span_id: Unique ID for this span
        parent_span_id: ID of parent span (if any)
    """
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid4())[:16])
    parent_span_id: str | None = None


@dataclass
class Span:
    """A single span in a trace.

    Represents a unit of work in the pipeline.

    Attributes:
        name: Name of the span (e.g., step name)
        context: Span context with IDs
        start_time: When the span started
        end_time: When the span ended
        status: Span status
        attributes: Key-value attributes
        events: List of events during the span
        error: Error information if failed
    """
    name: str
    context: SpanContext = field(default_factory=SpanContext)
    start_time: datetime = field(default_factory=_utc_now)
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": _utc_now().isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, error: str | None = None) -> None:
        """Set span status."""
        self.status = status
        if error:
            self.error = error

    def end(self, status: SpanStatus | None = None) -> None:
        """End the span."""
        self.end_time = _utc_now()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Serialize span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


class TraceExporter(ABC):
    """Abstract base for trace exporters."""

    @abstractmethod
    def export(self, spans: list[Span]) -> None:
        """Export spans to a backend."""
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleExporter(TraceExporter):
    """Export traces to console for debugging."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def export(self, spans: list[Span]) -> None:
        """Print spans to console."""
        for span in spans:
            status_icon = "✓" if span.status == SpanStatus.OK else "✗"
            print(f"[TRACE] {status_icon} {span.name} ({span.duration_ms:.2f}ms)")

            if self._verbose:
                print(f"        trace_id: {span.context.trace_id}")
                print(f"        span_id: {span.context.span_id}")
                if span.attributes:
                    print(f"        attributes: {span.attributes}")
                if span.error:
                    print(f"        error: {span.error}")


class InMemoryExporter(TraceExporter):
    """Export traces to memory for testing."""

    def __init__(self):
        self._spans: list[Span] = []

    def export(self, spans: list[Span]) -> None:
        """Store spans in memory."""
        self._spans.extend(spans)

    def get_spans(self) -> list[Span]:
        """Get all exported spans."""
        return self._spans.copy()

    def clear(self) -> None:
        """Clear stored spans."""
        self._spans.clear()


class Tracer:
    """Main tracer for RAG OS pipelines.

    Provides distributed tracing capabilities for tracking
    pipeline execution across steps.
    """

    def __init__(
        self,
        service_name: str = "rag-os",
        exporters: list[TraceExporter] | None = None,
    ):
        self._service_name = service_name
        self._exporters = exporters or []
        self._current_span: threading.local = threading.local()
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def get_current_span(self) -> Span | None:
        """Get the current active span."""
        return getattr(self._current_span, "span", None)

    def start_span(
        self,
        name: str,
        parent: Span | SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name
            parent: Parent span or context
            attributes: Initial attributes

        Returns:
            New span
        """
        # Determine parent context
        if parent is None:
            parent = self.get_current_span()

        if isinstance(parent, Span):
            context = SpanContext(
                trace_id=parent.context.trace_id,
                parent_span_id=parent.context.span_id,
            )
        elif isinstance(parent, SpanContext):
            context = SpanContext(
                trace_id=parent.trace_id,
                parent_span_id=parent.span_id,
            )
        else:
            context = SpanContext()

        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )

        span.set_attribute("service.name", self._service_name)

        return span

    @contextmanager
    def trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        """Context manager for tracing a block of code.

        Usage:
            with tracer.trace("my_operation") as span:
                span.set_attribute("key", "value")
                # do work
        """
        span = self.start_span(name, attributes=attributes)
        old_span = self.get_current_span()
        self._current_span.span = span

        try:
            yield span
            span.end(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.end()
            raise
        finally:
            self._current_span.span = old_span
            self._record_span(span)

    def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        with self._lock:
            self._spans.append(span)

        # Export if we have exporters
        if self._exporters:
            for exporter in self._exporters:
                exporter.export([span])

    def get_recorded_spans(self) -> list[Span]:
        """Get all recorded spans."""
        with self._lock:
            return self._spans.copy()

    def clear(self) -> None:
        """Clear recorded spans."""
        with self._lock:
            self._spans.clear()


# Global tracer instance
_global_tracer: Tracer | None = None


def get_tracer(
    service_name: str = "rag-os",
    exporters: list[TraceExporter] | None = None,
) -> Tracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(service_name, exporters)
    return _global_tracer
