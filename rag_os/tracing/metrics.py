"""Metrics collection for RAG OS."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from threading import Lock
from enum import Enum
import math


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class MetricType(Enum):
    """Type of metric."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=_utc_now)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """A counter metric that only increases.

    Good for tracking counts of events, requests, errors, etc.
    """

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = {}
        self._lock = Lock()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment the counter.

        Args:
            value: Amount to increment (must be positive)
            **labels: Label values
        """
        if value < 0:
            raise ValueError("Counter can only increase")

        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + value

    def get(self, **labels: str) -> float:
        """Get current counter value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)

    def _labels_to_key(self, labels: dict[str, str]) -> tuple:
        """Convert labels to hashable key."""
        return tuple(sorted(labels.items()))

    def collect(self) -> list[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(k))
                for k, v in self._values.items()
            ]


class Gauge:
    """A gauge metric that can go up or down.

    Good for tracking current values like queue size, active connections, etc.
    """

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = {}
        self._lock = Lock()

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment the gauge."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement the gauge."""
        self.inc(-value, **labels)

    def get(self, **labels: str) -> float:
        """Get current gauge value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)

    def _labels_to_key(self, labels: dict[str, str]) -> tuple:
        """Convert labels to hashable key."""
        return tuple(sorted(labels.items()))

    def collect(self) -> list[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(k))
                for k, v in self._values.items()
            ]


class Histogram:
    """A histogram metric for tracking distributions.

    Good for tracking latencies, sizes, etc.
    """

    # Default buckets for latency in milliseconds
    DEFAULT_BUCKETS = (5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 10000, float("inf"))

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._data: dict[tuple, dict[str, Any]] = {}
        self._lock = Lock()

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        label_key = self._labels_to_key(labels)

        with self._lock:
            if label_key not in self._data:
                self._data[label_key] = {
                    "count": 0,
                    "sum": 0.0,
                    "buckets": {b: 0 for b in self.buckets},
                    "values": [],
                }

            data = self._data[label_key]
            data["count"] += 1
            data["sum"] += value
            data["values"].append(value)

            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def get_count(self, **labels: str) -> int:
        """Get observation count."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            data = self._data.get(label_key)
            return data["count"] if data else 0

    def get_sum(self, **labels: str) -> float:
        """Get sum of observations."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            data = self._data.get(label_key)
            return data["sum"] if data else 0.0

    def get_mean(self, **labels: str) -> float:
        """Get mean of observations."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            data = self._data.get(label_key)
            if not data or data["count"] == 0:
                return 0.0
            return data["sum"] / data["count"]

    def get_percentile(self, percentile: float, **labels: str) -> float:
        """Get percentile value (0-100)."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            data = self._data.get(label_key)
            if not data or not data["values"]:
                return 0.0

            sorted_values = sorted(data["values"])
            idx = int(percentile / 100 * len(sorted_values))
            idx = min(idx, len(sorted_values) - 1)
            return sorted_values[idx]

    def _labels_to_key(self, labels: dict[str, str]) -> tuple:
        """Convert labels to hashable key."""
        return tuple(sorted(labels.items()))

    def collect(self) -> list[dict[str, Any]]:
        """Collect histogram data."""
        with self._lock:
            results = []
            for label_key, data in self._data.items():
                results.append({
                    "labels": dict(label_key),
                    "count": data["count"],
                    "sum": data["sum"],
                    "mean": data["sum"] / data["count"] if data["count"] > 0 else 0,
                    "buckets": data["buckets"].copy(),
                })
            return results


class MetricsCollector:
    """Central metrics collector for RAG OS.

    Provides pre-defined metrics for common RAG operations.
    """

    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = Lock()

        # Pre-define common RAG metrics
        self.pipeline_executions = self.counter(
            "rag_pipeline_executions_total",
            "Total pipeline executions",
            ["pipeline_name", "status"],
        )

        self.step_executions = self.counter(
            "rag_step_executions_total",
            "Total step executions",
            ["step_type", "step_id", "status"],
        )

        self.step_latency = self.histogram(
            "rag_step_latency_ms",
            "Step execution latency in milliseconds",
            ["step_type", "step_id"],
        )

        self.token_usage = self.counter(
            "rag_token_usage_total",
            "Total tokens used",
            ["step_id", "token_type"],
        )

        self.documents_processed = self.counter(
            "rag_documents_processed_total",
            "Total documents processed",
            ["step_type"],
        )

        self.chunks_created = self.counter(
            "rag_chunks_created_total",
            "Total chunks created",
        )

        self.retrieval_results = self.histogram(
            "rag_retrieval_results_count",
            "Number of results returned by retrieval",
            ["retriever_type"],
            buckets=(0, 1, 5, 10, 20, 50, 100, float("inf")),
        )

        self.active_pipelines = self.gauge(
            "rag_active_pipelines",
            "Currently running pipelines",
        )

    def counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Counter:
        """Create or get a counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description, labels)
            return self._metrics[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Gauge:
        """Create or get a gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description, labels)
            return self._metrics[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Create or get a histogram metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, labels, buckets)
            return self._metrics[name]

    def collect_all(self) -> dict[str, Any]:
        """Collect all metrics."""
        with self._lock:
            result = {}
            for name, metric in self._metrics.items():
                result[name] = metric.collect()
            return result


# Global metrics collector
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
