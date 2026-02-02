"""Evaluation metrics for RAG OS."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import re
import math


@dataclass
class MetricResult:
    """Result of computing a metric.

    Attributes:
        name: Name of the metric
        score: Numeric score (typically 0.0 to 1.0)
        details: Additional details about the computation
    """
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


class EvalMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """Compute the metric.

        Args:
            **kwargs: Metric-specific arguments

        Returns:
            MetricResult with score and details
        """
        pass


class RetrievalMetrics(EvalMetric):
    """Metrics for evaluating retrieval quality.

    Computes precision, recall, F1, MRR, and NDCG.
    """

    @property
    def name(self) -> str:
        return "retrieval_metrics"

    def compute(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int | None = None,
    ) -> MetricResult:
        """Compute retrieval metrics.

        Args:
            retrieved_ids: IDs of retrieved documents (in order)
            relevant_ids: IDs of actually relevant documents
            k: Consider only top-k retrieved (None = all)

        Returns:
            MetricResult with precision, recall, F1, MRR, NDCG
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        # Precision and Recall
        true_positives = len(retrieved_set & relevant_set)

        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0.0

        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        # NDCG (Normalized Discounted Cumulative Gain)
        ndcg = self._compute_ndcg(retrieved_ids, relevant_set)

        # Overall score is average of key metrics
        score = (precision + recall + f1 + mrr + ndcg) / 5

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": mrr,
                "ndcg": ndcg,
                "retrieved_count": len(retrieved_ids),
                "relevant_count": len(relevant_ids),
                "true_positives": true_positives,
            },
        )

    def _compute_ndcg(self, retrieved_ids: list[str], relevant_set: set[str]) -> float:
        """Compute NDCG score."""
        if not retrieved_ids or not relevant_set:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed

        # Ideal DCG (all relevant docs at the top)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), len(retrieved_ids))))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


class AnswerMetrics(EvalMetric):
    """Metrics for evaluating answer quality.

    Computes exact match, F1, and contains metrics.
    """

    @property
    def name(self) -> str:
        return "answer_metrics"

    def compute(
        self,
        predicted: str,
        expected: str,
        case_sensitive: bool = False,
    ) -> MetricResult:
        """Compute answer metrics.

        Args:
            predicted: Predicted answer
            expected: Expected answer
            case_sensitive: Whether to do case-sensitive comparison

        Returns:
            MetricResult with exact match, F1, and contains scores
        """
        if not case_sensitive:
            predicted = predicted.lower()
            expected = expected.lower()

        # Exact match
        exact_match = 1.0 if predicted.strip() == expected.strip() else 0.0

        # Contains (expected in predicted or vice versa)
        contains = 1.0 if expected in predicted or predicted in expected else 0.0

        # Token F1
        pred_tokens = set(self._tokenize(predicted))
        exp_tokens = set(self._tokenize(expected))

        common = pred_tokens & exp_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(exp_tokens) if exp_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Overall score is weighted average
        score = 0.4 * exact_match + 0.4 * f1 + 0.2 * contains

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "exact_match": exact_match,
                "f1": f1,
                "contains": contains,
                "token_precision": precision,
                "token_recall": recall,
            },
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization by splitting on non-alphanumeric."""
        return re.findall(r'\w+', text.lower())


class FaithfulnessMetric(EvalMetric):
    """Metric for evaluating faithfulness to source context.

    Checks if the answer can be supported by the provided context.
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(
        self,
        answer: str,
        context: str,
        strict: bool = False,
    ) -> MetricResult:
        """Compute faithfulness score.

        Args:
            answer: Generated answer
            context: Source context
            strict: Whether to use strict matching

        Returns:
            MetricResult with faithfulness score
        """
        if not answer or not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "Empty answer or context"},
            )

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Extract claims from answer (sentences)
        claims = [s.strip() for s in answer.split('.') if s.strip()]

        if not claims:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "No claims extracted"},
            )

        # Check each claim
        supported_claims = 0
        claim_details = []

        for claim in claims:
            # Extract key terms from claim
            terms = set(re.findall(r'\b\w{4,}\b', claim.lower()))

            if not terms:
                continue

            # Check how many terms appear in context
            terms_in_context = sum(1 for term in terms if term in context_lower)
            coverage = terms_in_context / len(terms) if terms else 0.0

            # Consider claim supported if coverage is above threshold
            threshold = 0.7 if strict else 0.5
            is_supported = coverage >= threshold

            if is_supported:
                supported_claims += 1

            claim_details.append({
                "claim": claim[:100],
                "coverage": coverage,
                "supported": is_supported,
            })

        # Faithfulness is ratio of supported claims
        score = supported_claims / len(claims) if claims else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "total_claims": len(claims),
                "supported_claims": supported_claims,
                "claim_analysis": claim_details[:5],  # Limit details
            },
        )


class RelevanceMetric(EvalMetric):
    """Metric for evaluating answer relevance to the question.

    Checks if the answer addresses the question.
    """

    @property
    def name(self) -> str:
        return "relevance"

    def compute(
        self,
        question: str,
        answer: str,
    ) -> MetricResult:
        """Compute relevance score.

        Args:
            question: The question asked
            answer: The generated answer

        Returns:
            MetricResult with relevance score
        """
        if not question or not answer:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "Empty question or answer"},
            )

        question_lower = question.lower()
        answer_lower = answer.lower()

        # Extract question terms
        question_terms = set(re.findall(r'\b\w{3,}\b', question_lower))

        # Remove common question words
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'the', 'is', 'are', 'was', 'were', 'does', 'did', 'can', 'could', 'would', 'should'}
        question_terms -= stop_words

        if not question_terms:
            return MetricResult(
                name=self.name,
                score=0.5,  # Neutral if no meaningful terms
                details={"reason": "No meaningful question terms"},
            )

        # Check term overlap
        terms_in_answer = sum(1 for term in question_terms if term in answer_lower)
        term_coverage = terms_in_answer / len(question_terms) if question_terms else 0.0

        # Check for refusal indicators
        refusal_phrases = [
            "i don't know",
            "i cannot",
            "i can't",
            "not available",
            "no information",
        ]
        has_refusal = any(phrase in answer_lower for phrase in refusal_phrases)

        # Check answer length (very short might be low quality)
        length_score = min(1.0, len(answer.split()) / 10)

        # Calculate overall relevance
        if has_refusal:
            score = 0.3  # Refusals get low but non-zero score
        else:
            score = 0.5 * term_coverage + 0.3 * length_score + 0.2

        score = max(0.0, min(1.0, score))

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "term_coverage": term_coverage,
                "length_score": length_score,
                "has_refusal": has_refusal,
                "question_terms": list(question_terms)[:10],
            },
        )


class ContextRecallMetric(EvalMetric):
    """Metric for evaluating context recall.

    Measures how much of the expected answer's information
    appears in the retrieved context.
    """

    @property
    def name(self) -> str:
        return "context_recall"

    def compute(
        self,
        context: str,
        expected_answer: str,
    ) -> MetricResult:
        """Compute context recall.

        Args:
            context: Retrieved context
            expected_answer: Expected answer

        Returns:
            MetricResult with recall score
        """
        if not context or not expected_answer:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "Empty context or expected answer"},
            )

        context_lower = context.lower()

        # Extract key terms from expected answer
        answer_terms = set(re.findall(r'\b\w{4,}\b', expected_answer.lower()))

        if not answer_terms:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": "No key terms in expected answer"},
            )

        # Count terms found in context
        found_terms = sum(1 for term in answer_terms if term in context_lower)
        recall = found_terms / len(answer_terms) if answer_terms else 0.0

        return MetricResult(
            name=self.name,
            score=recall,
            details={
                "total_terms": len(answer_terms),
                "found_terms": found_terms,
                "sample_terms": list(answer_terms)[:10],
            },
        )


class AnswerCorrectnessMetric(EvalMetric):
    """Metric for evaluating overall answer correctness.

    Combines multiple factors to assess correctness.
    """

    @property
    def name(self) -> str:
        return "answer_correctness"

    def __init__(self):
        self._answer_metrics = AnswerMetrics()
        self._faithfulness = FaithfulnessMetric()
        self._relevance = RelevanceMetric()

    def compute(
        self,
        question: str,
        predicted: str,
        expected: str,
        context: str | None = None,
    ) -> MetricResult:
        """Compute answer correctness.

        Args:
            question: The question
            predicted: Predicted answer
            expected: Expected answer
            context: Source context (optional)

        Returns:
            MetricResult with correctness score
        """
        # Answer similarity
        answer_result = self._answer_metrics.compute(predicted, expected)

        # Relevance to question
        relevance_result = self._relevance.compute(question, predicted)

        # Faithfulness (if context provided)
        if context:
            faithfulness_result = self._faithfulness.compute(predicted, context)
            faithfulness_score = faithfulness_result.score
        else:
            faithfulness_score = 1.0  # Assume faithful if no context

        # Weighted combination
        score = (
            0.5 * answer_result.score +
            0.3 * relevance_result.score +
            0.2 * faithfulness_score
        )

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "answer_similarity": answer_result.score,
                "relevance": relevance_result.score,
                "faithfulness": faithfulness_score,
                "answer_details": answer_result.details,
                "relevance_details": relevance_result.details,
            },
        )
