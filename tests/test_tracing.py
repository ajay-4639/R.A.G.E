"""Tests for RAG OS tracing and observability."""

import pytest
import time
from datetime import datetime, timezone
from io import StringIO
import sys

from rag_os.tracing import (
    Tracer,
    Span,
    SpanContext,
    TraceExporter,
    ConsoleExporter,
    get_tracer,
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
)
from rag_os.tracing.tracer import SpanStatus, InMemoryExporter


# =============================================================================
# SpanContext Tests
# =============================================================================

class TestSpanContext:
    """Tests for SpanContext."""

    def test_create_context(self):
        """Test creating a span context."""
        context = SpanContext()

        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.parent_span_id is None

    def test_context_with_parent(self):
        """Test context with parent span ID."""
        context = SpanContext(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="span-parent",
        )

        assert context.trace_id == "trace-123"
        assert context.span_id == "span-456"
        assert context.parent_span_id == "span-parent"


# =============================================================================
# Span Tests
# =============================================================================

class TestSpan:
    """Tests for Span."""

    def test_create_span(self):
        """Test creating a span."""
        span = Span(name="test_operation")

        assert span.name == "test_operation"
        assert span.context is not None
        assert span.start_time is not None
        assert span.end_time is None
        assert span.status == SpanStatus.UNSET
        assert span.attributes == {}
        assert span.events == []
        assert span.error is None

    def test_span_duration(self):
        """Test span duration calculation."""
        span = Span(name="test")

        # Duration should be 0 before ending
        assert span.duration_ms == 0.0

        time.sleep(0.1)
        span.end()

        # Duration should be > 0 after ending
        assert span.duration_ms >= 100

    def test_set_attribute(self):
        """Test setting span attributes."""
        span = Span(name="test")

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 123)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123

    def test_add_event(self):
        """Test adding events to span."""
        span = Span(name="test")

        span.add_event("event1", {"data": "value"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0]["name"] == "event1"
        assert span.events[0]["attributes"] == {"data": "value"}
        assert span.events[1]["name"] == "event2"

    def test_set_status_ok(self):
        """Test setting OK status."""
        span = Span(name="test")

        span.set_status(SpanStatus.OK)

        assert span.status == SpanStatus.OK
        assert span.error is None

    def test_set_status_error(self):
        """Test setting error status."""
        span = Span(name="test")

        span.set_status(SpanStatus.ERROR, "Something went wrong")

        assert span.status == SpanStatus.ERROR
        assert span.error == "Something went wrong"

    def test_end_span(self):
        """Test ending a span."""
        span = Span(name="test")

        span.end()

        assert span.end_time is not None
        assert span.status == SpanStatus.OK  # Default status when UNSET

    def test_end_span_with_status(self):
        """Test ending span with specific status."""
        span = Span(name="test")

        span.end(SpanStatus.ERROR)

        assert span.status == SpanStatus.ERROR

    def test_to_dict(self):
        """Test serializing span to dictionary."""
        span = Span(name="test")
        span.set_attribute("key", "value")
        span.add_event("event1")
        span.end()

        data = span.to_dict()

        assert data["name"] == "test"
        assert data["trace_id"] is not None
        assert data["span_id"] is not None
        assert data["start_time"] is not None
        assert data["end_time"] is not None
        assert data["duration_ms"] >= 0
        assert data["status"] == "ok"
        assert data["attributes"]["key"] == "value"
        assert len(data["events"]) == 1


# =============================================================================
# TraceExporter Tests
# =============================================================================

class TestInMemoryExporter:
    """Tests for InMemoryExporter."""

    def test_export_spans(self):
        """Test exporting spans to memory."""
        exporter = InMemoryExporter()
        span = Span(name="test")
        span.end()

        exporter.export([span])

        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "test"

    def test_get_spans_returns_copy(self):
        """Test that get_spans returns a copy."""
        exporter = InMemoryExporter()
        span = Span(name="test")
        exporter.export([span])

        spans1 = exporter.get_spans()
        spans2 = exporter.get_spans()

        assert spans1 is not spans2

    def test_clear(self):
        """Test clearing exported spans."""
        exporter = InMemoryExporter()
        span = Span(name="test")
        exporter.export([span])

        exporter.clear()

        assert len(exporter.get_spans()) == 0


class TestConsoleExporter:
    """Tests for ConsoleExporter."""

    def test_export_to_console(self, capsys):
        """Test exporting spans to console."""
        exporter = ConsoleExporter()
        span = Span(name="test_operation")
        span.end()

        exporter.export([span])

        captured = capsys.readouterr()
        assert "[TRACE]" in captured.out
        assert "test_operation" in captured.out

    def test_export_verbose(self, capsys):
        """Test verbose console output."""
        exporter = ConsoleExporter(verbose=True)
        span = Span(name="test_operation")
        span.set_attribute("key", "value")
        span.end()

        exporter.export([span])

        captured = capsys.readouterr()
        assert "trace_id" in captured.out
        assert "span_id" in captured.out


# =============================================================================
# Tracer Tests
# =============================================================================

class TestTracer:
    """Tests for Tracer."""

    def test_create_tracer(self):
        """Test creating a tracer."""
        tracer = Tracer(service_name="test-service")

        assert tracer._service_name == "test-service"

    def test_start_span(self):
        """Test starting a span."""
        tracer = Tracer()

        span = tracer.start_span("test_operation")

        assert span.name == "test_operation"
        assert span.attributes["service.name"] == "rag-os"

    def test_start_span_with_attributes(self):
        """Test starting span with attributes."""
        tracer = Tracer()

        span = tracer.start_span("test", attributes={"key": "value"})

        assert span.attributes["key"] == "value"

    def test_start_span_with_parent(self):
        """Test starting span with parent."""
        tracer = Tracer()

        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)

        assert child.context.parent_span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id

    def test_trace_context_manager(self):
        """Test trace context manager."""
        tracer = Tracer()

        with tracer.trace("test_operation") as span:
            span.set_attribute("key", "value")

        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_trace_context_manager_error(self):
        """Test trace context manager with error."""
        tracer = Tracer()

        with pytest.raises(ValueError):
            with tracer.trace("test_operation") as span:
                raise ValueError("test error")

        assert span.status == SpanStatus.ERROR
        assert "test error" in span.error

    def test_nested_traces(self):
        """Test nested trace contexts."""
        tracer = Tracer()

        with tracer.trace("parent") as parent_span:
            with tracer.trace("child") as child_span:
                pass

        # Child should have parent's trace ID
        assert child_span.context.trace_id == parent_span.context.trace_id
        # Child should reference parent
        assert child_span.context.parent_span_id == parent_span.context.span_id

    def test_get_current_span(self):
        """Test getting current span."""
        tracer = Tracer()

        # No current span initially
        assert tracer.get_current_span() is None

        with tracer.trace("test") as span:
            # Current span should be set
            assert tracer.get_current_span() == span

        # No current span after context
        assert tracer.get_current_span() is None

    def test_get_recorded_spans(self):
        """Test getting recorded spans."""
        tracer = Tracer()

        with tracer.trace("span1"):
            pass
        with tracer.trace("span2"):
            pass

        spans = tracer.get_recorded_spans()

        assert len(spans) == 2
        names = [s.name for s in spans]
        assert "span1" in names
        assert "span2" in names

    def test_clear_recorded_spans(self):
        """Test clearing recorded spans."""
        tracer = Tracer()

        with tracer.trace("test"):
            pass

        tracer.clear()

        assert len(tracer.get_recorded_spans()) == 0

    def test_tracer_with_exporter(self):
        """Test tracer with exporter."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter])

        with tracer.trace("test"):
            pass

        assert len(exporter.get_spans()) == 1


# =============================================================================
# Global Tracer Tests
# =============================================================================

class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_singleton(self):
        """Test that get_tracer returns singleton."""
        # Note: In a real test we'd reset the global, but this tests the basic behavior
        tracer1 = get_tracer()
        tracer2 = get_tracer()

        assert tracer1 is tracer2


# =============================================================================
# Counter Tests
# =============================================================================

class TestCounter:
    """Tests for Counter metric."""

    def test_create_counter(self):
        """Test creating a counter."""
        counter = Counter("test_counter", "Test counter")

        assert counter.name == "test_counter"
        assert counter.description == "Test counter"

    def test_increment(self):
        """Test incrementing counter."""
        counter = Counter("test")

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5.0)
        assert counter.get() == 6.0

    def test_increment_negative_raises(self):
        """Test that negative increment raises error."""
        counter = Counter("test")

        with pytest.raises(ValueError, match="Counter can only increase"):
            counter.inc(-1.0)

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter("requests", labels=["method", "status"])

        counter.inc(method="GET", status="200")
        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="201")

        assert counter.get(method="GET", status="200") == 2.0
        assert counter.get(method="POST", status="201") == 1.0
        assert counter.get(method="DELETE", status="404") == 0.0

    def test_collect(self):
        """Test collecting counter values."""
        counter = Counter("test", labels=["label"])
        counter.inc(1.0, label="a")
        counter.inc(2.0, label="b")

        values = counter.collect()

        assert len(values) == 2


# =============================================================================
# Gauge Tests
# =============================================================================

class TestGauge:
    """Tests for Gauge metric."""

    def test_create_gauge(self):
        """Test creating a gauge."""
        gauge = Gauge("test_gauge", "Test gauge")

        assert gauge.name == "test_gauge"
        assert gauge.description == "Test gauge"

    def test_set(self):
        """Test setting gauge value."""
        gauge = Gauge("test")

        gauge.set(10.0)
        assert gauge.get() == 10.0

        gauge.set(5.0)
        assert gauge.get() == 5.0

    def test_increment(self):
        """Test incrementing gauge."""
        gauge = Gauge("test")

        gauge.inc()
        assert gauge.get() == 1.0

        gauge.inc(5.0)
        assert gauge.get() == 6.0

    def test_decrement(self):
        """Test decrementing gauge."""
        gauge = Gauge("test")
        gauge.set(10.0)

        gauge.dec()
        assert gauge.get() == 9.0

        gauge.dec(4.0)
        assert gauge.get() == 5.0

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("connections", labels=["type"])

        gauge.set(10.0, type="websocket")
        gauge.set(5.0, type="http")

        assert gauge.get(type="websocket") == 10.0
        assert gauge.get(type="http") == 5.0

    def test_collect(self):
        """Test collecting gauge values."""
        gauge = Gauge("test", labels=["label"])
        gauge.set(1.0, label="a")
        gauge.set(2.0, label="b")

        values = gauge.collect()

        assert len(values) == 2


# =============================================================================
# Histogram Tests
# =============================================================================

class TestHistogram:
    """Tests for Histogram metric."""

    def test_create_histogram(self):
        """Test creating a histogram."""
        histogram = Histogram("test_histogram", "Test histogram")

        assert histogram.name == "test_histogram"
        assert histogram.description == "Test histogram"
        assert histogram.buckets == Histogram.DEFAULT_BUCKETS

    def test_custom_buckets(self):
        """Test histogram with custom buckets."""
        buckets = (10, 50, 100, float("inf"))
        histogram = Histogram("test", buckets=buckets)

        assert histogram.buckets == buckets

    def test_observe(self):
        """Test observing values."""
        histogram = Histogram("test")

        histogram.observe(10.0)
        histogram.observe(20.0)
        histogram.observe(30.0)

        assert histogram.get_count() == 3
        assert histogram.get_sum() == 60.0

    def test_get_mean(self):
        """Test getting mean."""
        histogram = Histogram("test")

        histogram.observe(10.0)
        histogram.observe(20.0)
        histogram.observe(30.0)

        assert histogram.get_mean() == 20.0

    def test_get_mean_empty(self):
        """Test getting mean of empty histogram."""
        histogram = Histogram("test")

        assert histogram.get_mean() == 0.0

    def test_get_percentile(self):
        """Test getting percentiles."""
        histogram = Histogram("test")

        for i in range(1, 101):
            histogram.observe(float(i))

        # 50th percentile should be around 50
        p50 = histogram.get_percentile(50)
        assert 49 <= p50 <= 51

        # 90th percentile should be around 90
        p90 = histogram.get_percentile(90)
        assert 89 <= p90 <= 91

    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        histogram = Histogram("latency", labels=["endpoint"])

        histogram.observe(10.0, endpoint="/api/users")
        histogram.observe(20.0, endpoint="/api/users")
        histogram.observe(100.0, endpoint="/api/orders")

        assert histogram.get_count(endpoint="/api/users") == 2
        assert histogram.get_count(endpoint="/api/orders") == 1
        assert histogram.get_mean(endpoint="/api/users") == 15.0

    def test_collect(self):
        """Test collecting histogram data."""
        histogram = Histogram("test", labels=["label"])
        histogram.observe(10.0, label="a")
        histogram.observe(20.0, label="a")
        histogram.observe(30.0, label="b")

        data = histogram.collect()

        assert len(data) == 2


# =============================================================================
# MetricsCollector Tests
# =============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_create_collector(self):
        """Test creating metrics collector."""
        collector = MetricsCollector()

        # Should have pre-defined metrics
        assert collector.pipeline_executions is not None
        assert collector.step_executions is not None
        assert collector.step_latency is not None
        assert collector.token_usage is not None

    def test_create_counter(self):
        """Test creating a counter via collector."""
        collector = MetricsCollector()

        counter = collector.counter("my_counter", "Description")
        counter.inc()

        assert counter.get() == 1.0

    def test_create_gauge(self):
        """Test creating a gauge via collector."""
        collector = MetricsCollector()

        gauge = collector.gauge("my_gauge", "Description")
        gauge.set(10.0)

        assert gauge.get() == 10.0

    def test_create_histogram(self):
        """Test creating a histogram via collector."""
        collector = MetricsCollector()

        histogram = collector.histogram("my_histogram", "Description")
        histogram.observe(100.0)

        assert histogram.get_count() == 1

    def test_get_existing_metric(self):
        """Test getting existing metric by name."""
        collector = MetricsCollector()

        counter1 = collector.counter("my_counter")
        counter1.inc()

        counter2 = collector.counter("my_counter")

        # Should return same counter
        assert counter2.get() == 1.0

    def test_collect_all(self):
        """Test collecting all metrics."""
        collector = MetricsCollector()

        # Use some metrics
        collector.pipeline_executions.inc(pipeline_name="test", status="success")
        collector.active_pipelines.set(5)

        all_metrics = collector.collect_all()

        assert "rag_pipeline_executions_total" in all_metrics
        assert "rag_active_pipelines" in all_metrics

    def test_predefined_metrics(self):
        """Test using pre-defined RAG metrics."""
        collector = MetricsCollector()

        # Track pipeline execution
        collector.pipeline_executions.inc(pipeline_name="qa", status="success")
        collector.pipeline_executions.inc(pipeline_name="qa", status="success")
        collector.pipeline_executions.inc(pipeline_name="qa", status="error")

        assert collector.pipeline_executions.get(pipeline_name="qa", status="success") == 2.0
        assert collector.pipeline_executions.get(pipeline_name="qa", status="error") == 1.0

        # Track step latency
        collector.step_latency.observe(50.0, step_type="embedding", step_id="embed_1")
        collector.step_latency.observe(100.0, step_type="embedding", step_id="embed_1")

        assert collector.step_latency.get_count(step_type="embedding", step_id="embed_1") == 2
        assert collector.step_latency.get_mean(step_type="embedding", step_id="embed_1") == 75.0

        # Track active pipelines
        collector.active_pipelines.inc()
        collector.active_pipelines.inc()
        collector.active_pipelines.dec()

        assert collector.active_pipelines.get() == 1.0
