"""Judge implementations for RAG evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import re


@dataclass
class JudgmentResult:
    """Result of a judgment.

    Attributes:
        score: Numeric score (0.0 to 1.0)
        passed: Whether the judgment passed
        reason: Explanation for the judgment
        details: Additional details
    """
    score: float
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class BaseJudge(ABC):
    """Abstract base class for judges."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the judge."""
        pass

    @abstractmethod
    def judge(
        self,
        question: str,
        answer: str,
        expected: str | None = None,
        context: str | None = None,
    ) -> JudgmentResult:
        """Judge an answer.

        Args:
            question: The question asked
            answer: The generated answer
            expected: Expected answer (optional)
            context: Source context (optional)

        Returns:
            JudgmentResult with score and explanation
        """
        pass


class RuleBasedJudge(BaseJudge):
    """Judge based on configurable rules.

    Applies a series of rules to evaluate answers.
    """

    @property
    def name(self) -> str:
        return "rule_based_judge"

    def __init__(
        self,
        rules: list[Callable[[str, str, str | None, str | None], tuple[float, str]]] | None = None,
        passing_threshold: float = 0.5,
    ):
        """Initialize with rules.

        Args:
            rules: List of rule functions that return (score, reason)
            passing_threshold: Minimum score to pass
        """
        self._rules = rules or self._default_rules()
        self._threshold = passing_threshold

    def _default_rules(self) -> list[Callable]:
        """Get default evaluation rules."""
        return [
            self._rule_not_empty,
            self._rule_min_length,
            self._rule_not_refusal,
            self._rule_contains_expected,
        ]

    def _rule_not_empty(
        self,
        question: str,
        answer: str,
        expected: str | None,
        context: str | None,
    ) -> tuple[float, str]:
        """Rule: Answer should not be empty."""
        if not answer or not answer.strip():
            return 0.0, "Answer is empty"
        return 1.0, "Answer is not empty"

    def _rule_min_length(
        self,
        question: str,
        answer: str,
        expected: str | None,
        context: str | None,
    ) -> tuple[float, str]:
        """Rule: Answer should have minimum length."""
        word_count = len(answer.split())
        if word_count < 3:
            return 0.3, f"Answer too short ({word_count} words)"
        if word_count < 10:
            return 0.7, f"Answer somewhat short ({word_count} words)"
        return 1.0, f"Answer has adequate length ({word_count} words)"

    def _rule_not_refusal(
        self,
        question: str,
        answer: str,
        expected: str | None,
        context: str | None,
    ) -> tuple[float, str]:
        """Rule: Answer should not be a refusal."""
        refusal_patterns = [
            r"i don'?t know",
            r"i cannot",
            r"i'?m not (sure|able)",
            r"no information",
            r"unable to",
        ]

        answer_lower = answer.lower()
        for pattern in refusal_patterns:
            if re.search(pattern, answer_lower):
                return 0.2, f"Answer appears to be a refusal"

        return 1.0, "Answer is not a refusal"

    def _rule_contains_expected(
        self,
        question: str,
        answer: str,
        expected: str | None,
        context: str | None,
    ) -> tuple[float, str]:
        """Rule: Answer should contain expected content."""
        if not expected:
            return 1.0, "No expected answer to compare"

        answer_lower = answer.lower()
        expected_lower = expected.lower()

        # Check for exact match
        if expected_lower in answer_lower:
            return 1.0, "Answer contains expected content"

        # Check for key terms
        expected_terms = set(re.findall(r'\b\w{4,}\b', expected_lower))
        if not expected_terms:
            return 0.5, "No meaningful terms in expected answer"

        found_terms = sum(1 for term in expected_terms if term in answer_lower)
        coverage = found_terms / len(expected_terms)

        if coverage >= 0.8:
            return 0.9, f"Answer has high term coverage ({coverage:.0%})"
        if coverage >= 0.5:
            return 0.6, f"Answer has moderate term coverage ({coverage:.0%})"
        return 0.3, f"Answer has low term coverage ({coverage:.0%})"

    def judge(
        self,
        question: str,
        answer: str,
        expected: str | None = None,
        context: str | None = None,
    ) -> JudgmentResult:
        """Apply all rules and aggregate."""
        rule_results = []
        total_score = 0.0

        for rule in self._rules:
            score, reason = rule(question, answer, expected, context)
            rule_results.append({"score": score, "reason": reason})
            total_score += score

        avg_score = total_score / len(self._rules) if self._rules else 0.0
        passed = avg_score >= self._threshold

        # Find main reason for score
        min_result = min(rule_results, key=lambda x: x["score"])
        main_reason = min_result["reason"] if min_result["score"] < 1.0 else "All rules passed"

        return JudgmentResult(
            score=avg_score,
            passed=passed,
            reason=main_reason,
            details={
                "rule_results": rule_results,
                "threshold": self._threshold,
            },
        )


class LLMJudge(BaseJudge):
    """Judge using an LLM for evaluation.

    Uses an LLM to evaluate answer quality.
    """

    @property
    def name(self) -> str:
        return "llm_judge"

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        passing_threshold: float = 0.7,
        criteria: list[str] | None = None,
    ):
        """Initialize LLM judge.

        Args:
            llm_fn: Function that calls LLM (takes prompt, returns response)
            passing_threshold: Minimum score to pass
            criteria: List of evaluation criteria
        """
        self._llm_fn = llm_fn
        self._threshold = passing_threshold
        self._criteria = criteria or [
            "relevance: Does the answer address the question?",
            "accuracy: Is the answer factually correct?",
            "completeness: Does the answer cover all aspects?",
            "coherence: Is the answer well-structured and clear?",
        ]

    def _build_prompt(
        self,
        question: str,
        answer: str,
        expected: str | None,
        context: str | None,
    ) -> str:
        """Build evaluation prompt."""
        criteria_text = "\n".join(f"- {c}" for c in self._criteria)

        prompt = f"""You are evaluating the quality of an answer to a question.

Question: {question}

Answer to evaluate: {answer}
"""

        if expected:
            prompt += f"\nExpected answer: {expected}"

        if context:
            prompt += f"\nContext: {context[:1000]}..."

        prompt += f"""

Evaluate the answer based on these criteria:
{criteria_text}

Provide your evaluation in this exact format:
SCORE: [number from 0.0 to 1.0]
REASON: [one sentence explanation]
"""

        return prompt

    def _parse_response(self, response: str) -> tuple[float, str]:
        """Parse LLM response for score and reason."""
        # Try to extract score
        score_match = re.search(r'SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5

        # Clamp score to valid range
        score = max(0.0, min(1.0, score))

        # Try to extract reason
        reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"

        return score, reason

    def judge(
        self,
        question: str,
        answer: str,
        expected: str | None = None,
        context: str | None = None,
    ) -> JudgmentResult:
        """Judge using LLM."""
        if not self._llm_fn:
            # Fall back to rule-based if no LLM
            return JudgmentResult(
                score=0.5,
                passed=True,
                reason="No LLM function provided, defaulting to neutral score",
                details={"fallback": True},
            )

        try:
            prompt = self._build_prompt(question, answer, expected, context)
            response = self._llm_fn(prompt)
            score, reason = self._parse_response(response)

            return JudgmentResult(
                score=score,
                passed=score >= self._threshold,
                reason=reason,
                details={
                    "prompt": prompt[:500],
                    "response": response[:500],
                    "criteria": self._criteria,
                },
            )

        except Exception as e:
            return JudgmentResult(
                score=0.0,
                passed=False,
                reason=f"LLM evaluation failed: {str(e)}",
                details={"error": str(e)},
            )


class CompositeJudge(BaseJudge):
    """Combines multiple judges."""

    @property
    def name(self) -> str:
        return "composite_judge"

    def __init__(
        self,
        judges: list[BaseJudge],
        weights: list[float] | None = None,
        passing_threshold: float = 0.6,
    ):
        """Initialize composite judge.

        Args:
            judges: List of judges to combine
            weights: Weights for each judge (default: equal)
            passing_threshold: Minimum score to pass
        """
        self._judges = judges
        self._weights = weights or [1.0 / len(judges)] * len(judges)
        self._threshold = passing_threshold

        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

    def judge(
        self,
        question: str,
        answer: str,
        expected: str | None = None,
        context: str | None = None,
    ) -> JudgmentResult:
        """Combine judgments from all judges."""
        results = []
        weighted_score = 0.0

        for judge, weight in zip(self._judges, self._weights):
            result = judge.judge(question, answer, expected, context)
            results.append({
                "judge": judge.name,
                "score": result.score,
                "reason": result.reason,
                "weight": weight,
            })
            weighted_score += result.score * weight

        passed = weighted_score >= self._threshold

        # Find the lowest-scoring judge for main reason
        worst = min(results, key=lambda x: x["score"])
        main_reason = f"{worst['judge']}: {worst['reason']}" if worst["score"] < 0.5 else "All judges passed"

        return JudgmentResult(
            score=weighted_score,
            passed=passed,
            reason=main_reason,
            details={
                "judge_results": results,
                "threshold": self._threshold,
            },
        )
