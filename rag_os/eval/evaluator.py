"""Evaluator for RAG OS pipelines."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4
import statistics

from rag_os.eval.metrics import EvalMetric, MetricResult


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class EvalSample:
    """A single evaluation sample.

    Attributes:
        id: Unique sample ID
        question: The question/query
        expected_answer: Expected answer (ground truth)
        context: Optional context documents
        metadata: Additional metadata
    """
    question: str
    expected_answer: str
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    context: list[str] = field(default_factory=list)
    expected_retrieved_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """A dataset for evaluation.

    Attributes:
        name: Dataset name
        samples: List of evaluation samples
        description: Dataset description
    """
    name: str
    samples: list[EvalSample]
    description: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def from_list(
        cls,
        name: str,
        data: list[dict[str, Any]],
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> "EvalDataset":
        """Create dataset from list of dicts.

        Args:
            name: Dataset name
            data: List of sample dicts
            question_key: Key for question field
            answer_key: Key for answer field

        Returns:
            EvalDataset
        """
        samples = []
        for item in data:
            sample = EvalSample(
                question=item[question_key],
                expected_answer=item[answer_key],
                context=item.get("context", []),
                expected_retrieved_ids=item.get("expected_retrieved_ids", []),
                metadata={k: v for k, v in item.items() if k not in [question_key, answer_key, "context", "expected_retrieved_ids"]},
            )
            samples.append(sample)

        return cls(name=name, samples=samples)


@dataclass
class EvalConfig:
    """Configuration for evaluation.

    Attributes:
        metrics: List of metrics to compute
        batch_size: Batch size for evaluation
        verbose: Whether to print progress
        fail_fast: Stop on first error
    """
    metrics: list[EvalMetric] = field(default_factory=list)
    batch_size: int = 10
    verbose: bool = False
    fail_fast: bool = False


@dataclass
class SampleResult:
    """Result for a single sample.

    Attributes:
        sample_id: Sample ID
        predicted: Predicted answer
        metrics: Metric results
        error: Error message if failed
        latency_ms: Time to generate answer
    """
    sample_id: str
    predicted: str
    metrics: dict[str, MetricResult]
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Check if all metrics passed threshold."""
        return self.error is None


@dataclass
class EvalResult:
    """Overall evaluation result.

    Attributes:
        dataset_name: Name of evaluated dataset
        sample_results: Results for each sample
        aggregate_metrics: Aggregated metric scores
        started_at: When evaluation started
        completed_at: When evaluation completed
    """
    dataset_name: str
    sample_results: list[SampleResult]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None

    @property
    def total_samples(self) -> int:
        return len(self.sample_results)

    @property
    def passed_samples(self) -> int:
        return sum(1 for r in self.sample_results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_samples / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "pass_rate": self.pass_rate,
            "aggregate_metrics": self.aggregate_metrics,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Evaluator:
    """Evaluates RAG pipeline responses.

    Usage:
        evaluator = Evaluator(config)
        result = evaluator.evaluate(dataset, answer_fn)
    """

    def __init__(self, config: EvalConfig | None = None):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvalConfig()

    def evaluate(
        self,
        dataset: EvalDataset,
        answer_fn: Callable[[str], str],
        retrieved_fn: Callable[[str], list[str]] | None = None,
    ) -> EvalResult:
        """Evaluate a dataset.

        Args:
            dataset: Dataset to evaluate
            answer_fn: Function that takes question and returns answer
            retrieved_fn: Optional function that returns retrieved doc IDs

        Returns:
            EvalResult with scores
        """
        import time

        result = EvalResult(
            dataset_name=dataset.name,
            sample_results=[],
        )

        for sample in dataset:
            sample_start = time.time()

            try:
                # Get prediction
                predicted = answer_fn(sample.question)

                # Get retrieved IDs if function provided
                retrieved_ids = []
                if retrieved_fn:
                    retrieved_ids = retrieved_fn(sample.question)

                # Compute metrics
                metric_results = self._compute_metrics(
                    sample=sample,
                    predicted=predicted,
                    retrieved_ids=retrieved_ids,
                )

                latency = (time.time() - sample_start) * 1000

                sample_result = SampleResult(
                    sample_id=sample.id,
                    predicted=predicted,
                    metrics=metric_results,
                    latency_ms=latency,
                )

            except Exception as e:
                sample_result = SampleResult(
                    sample_id=sample.id,
                    predicted="",
                    metrics={},
                    error=str(e),
                )

                if self.config.fail_fast:
                    raise

            result.sample_results.append(sample_result)

            if self.config.verbose:
                status = "✓" if sample_result.passed else "✗"
                print(f"[{status}] Sample {sample.id}: {sample_result.latency_ms:.1f}ms")

        # Compute aggregate metrics
        result.aggregate_metrics = self._aggregate_metrics(result.sample_results)
        result.completed_at = _utc_now()

        return result

    def _compute_metrics(
        self,
        sample: EvalSample,
        predicted: str,
        retrieved_ids: list[str],
    ) -> dict[str, MetricResult]:
        """Compute all configured metrics for a sample."""
        results = {}

        for metric in self.config.metrics:
            try:
                # Different metrics need different inputs
                if metric.name == "retrieval_metrics":
                    result = metric.compute(
                        retrieved_ids=retrieved_ids,
                        relevant_ids=sample.expected_retrieved_ids,
                    )
                elif metric.name == "answer_metrics":
                    result = metric.compute(
                        predicted=predicted,
                        expected=sample.expected_answer,
                    )
                elif metric.name == "faithfulness":
                    context = " ".join(sample.context) if sample.context else ""
                    result = metric.compute(
                        answer=predicted,
                        context=context,
                    )
                elif metric.name == "relevance":
                    result = metric.compute(
                        question=sample.question,
                        answer=predicted,
                    )
                elif metric.name == "context_recall":
                    context = " ".join(sample.context) if sample.context else ""
                    result = metric.compute(
                        context=context,
                        expected_answer=sample.expected_answer,
                    )
                elif metric.name == "answer_correctness":
                    context = " ".join(sample.context) if sample.context else None
                    result = metric.compute(
                        question=sample.question,
                        predicted=predicted,
                        expected=sample.expected_answer,
                        context=context,
                    )
                else:
                    # Generic call
                    result = metric.compute(
                        predicted=predicted,
                        expected=sample.expected_answer,
                    )

                results[metric.name] = result

            except Exception as e:
                results[metric.name] = MetricResult(
                    name=metric.name,
                    score=0.0,
                    details={"error": str(e)},
                )

        return results

    def _aggregate_metrics(
        self,
        sample_results: list[SampleResult],
    ) -> dict[str, float]:
        """Aggregate metrics across samples."""
        aggregated = {}

        # Collect all metric names
        metric_names = set()
        for sr in sample_results:
            metric_names.update(sr.metrics.keys())

        # Compute mean for each metric
        for name in metric_names:
            scores = [
                sr.metrics[name].score
                for sr in sample_results
                if name in sr.metrics
            ]
            if scores:
                aggregated[f"{name}_mean"] = statistics.mean(scores)
                if len(scores) > 1:
                    aggregated[f"{name}_std"] = statistics.stdev(scores)
                else:
                    aggregated[f"{name}_std"] = 0.0

        # Add latency stats
        latencies = [sr.latency_ms for sr in sample_results if sr.latency_ms > 0]
        if latencies:
            aggregated["latency_mean_ms"] = statistics.mean(latencies)
            aggregated["latency_p50_ms"] = statistics.median(latencies)
            if len(latencies) > 1:
                sorted_lat = sorted(latencies)
                p90_idx = int(len(sorted_lat) * 0.9)
                aggregated["latency_p90_ms"] = sorted_lat[p90_idx]

        return aggregated

    def compare(
        self,
        results: list[EvalResult],
    ) -> dict[str, Any]:
        """Compare multiple evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Comparison summary
        """
        if not results:
            return {}

        comparison = {
            "datasets": [r.dataset_name for r in results],
            "sample_counts": [r.total_samples for r in results],
            "pass_rates": [r.pass_rate for r in results],
        }

        # Compare common metrics
        common_metrics = set()
        for r in results:
            if not common_metrics:
                common_metrics = set(r.aggregate_metrics.keys())
            else:
                common_metrics &= set(r.aggregate_metrics.keys())

        for metric in common_metrics:
            comparison[metric] = [r.aggregate_metrics.get(metric, 0.0) for r in results]

        return comparison
