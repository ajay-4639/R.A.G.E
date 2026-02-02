"""Post-processing implementations for RAG OS."""

import re
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.steps.llm.base import LLMResponse
from rag_os.steps.post_processing.base import BasePostProcessingStep, ProcessedResponse


@register_step(
    name="CitationExtractionStep",
    step_type=StepType.POST_PROCESSING,
    description="Extract citations from response",
    version="1.0.0",
)
class CitationExtractionStep(BasePostProcessingStep):
    """Extract and validate citations from LLM responses."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._pattern = r'\[(\d+)\]'
        self._sources = config.get("sources", []) if config else []

    def process(self, response: str | LLMResponse) -> ProcessedResponse:
        """Extract citations from response."""
        content = response.content if isinstance(response, LLMResponse) else response

        # Find all citations
        citations = []
        matches = re.findall(self._pattern, content)

        for match in set(matches):
            source_idx = int(match) - 1  # Convert to 0-indexed
            citation = {"id": f"[{match}]", "index": source_idx}

            # Add source info if available
            if self._sources and 0 <= source_idx < len(self._sources):
                citation["source"] = self._sources[source_idx]

            citations.append(citation)

        # Sort by index
        citations.sort(key=lambda c: c["index"])

        return ProcessedResponse(
            content=content,
            original=content,
            citations=citations,
            metadata={"citation_count": len(citations)},
        )


@register_step(
    name="ResponseFormattingStep",
    step_type=StepType.POST_PROCESSING,
    description="Format and clean response text",
    version="1.0.0",
)
class ResponseFormattingStep(BasePostProcessingStep):
    """Format and clean LLM responses."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._strip_whitespace = config.get("strip_whitespace", True) if config else True
        self._remove_thinking = config.get("remove_thinking", True) if config else True
        self._max_length = config.get("max_length") if config else None

    def process(self, response: str | LLMResponse) -> ProcessedResponse:
        """Format the response."""
        content = response.content if isinstance(response, LLMResponse) else response
        original = content

        # Strip whitespace
        if self._strip_whitespace:
            content = content.strip()

        # Remove "thinking" or "chain of thought" markers
        if self._remove_thinking:
            # Remove <thinking>...</thinking> blocks
            content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
            # Remove common CoT patterns
            content = re.sub(r'Let me think.*?\.\s*', '', content, flags=re.IGNORECASE)

        # Truncate if needed
        if self._max_length and len(content) > self._max_length:
            content = content[:self._max_length] + "..."

        return ProcessedResponse(
            content=content,
            original=original,
            metadata={
                "formatted": True,
                "length_original": len(original),
                "length_final": len(content),
            },
        )


@register_step(
    name="AnswerValidationStep",
    step_type=StepType.POST_PROCESSING,
    description="Validate answer quality",
    version="1.0.0",
)
class AnswerValidationStep(BasePostProcessingStep):
    """Validate answer quality and completeness."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._min_length = config.get("min_length", 10) if config else 10
        self._require_sources = config.get("require_sources", False) if config else False
        self._blocked_phrases = config.get("blocked_phrases", []) if config else []

    def process(self, response: str | LLMResponse) -> ProcessedResponse:
        """Validate the response."""
        content = response.content if isinstance(response, LLMResponse) else response

        warnings = []
        confidence = 1.0

        # Check length
        if len(content) < self._min_length:
            warnings.append("Response is very short")
            confidence -= 0.2

        # Check for uncertainty indicators
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "i can't",
            "no information",
            "not mentioned",
        ]
        content_lower = content.lower()
        for phrase in uncertainty_phrases:
            if phrase in content_lower:
                warnings.append(f"Response indicates uncertainty: '{phrase}'")
                confidence -= 0.1

        # Check for sources if required
        if self._require_sources:
            if not re.search(r'\[\d+\]', content):
                warnings.append("No source citations found")
                confidence -= 0.15

        # Check for blocked phrases
        for phrase in self._blocked_phrases:
            if phrase.lower() in content_lower:
                warnings.append(f"Blocked phrase found: '{phrase}'")
                confidence -= 0.3

        # Ensure confidence stays in valid range
        confidence = max(0.0, min(1.0, confidence))

        return ProcessedResponse(
            content=content,
            original=content,
            confidence=confidence,
            warnings=warnings,
            metadata={"validation_passed": len(warnings) == 0},
        )


@register_step(
    name="HallucinationCheckStep",
    step_type=StepType.POST_PROCESSING,
    description="Check for potential hallucinations",
    version="1.0.0",
)
class HallucinationCheckStep(BasePostProcessingStep):
    """Check for potential hallucinations by comparing to context."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._context = config.get("context", "") if config else ""
        self._strict = config.get("strict", False) if config else False

    def process(self, response: str | LLMResponse) -> ProcessedResponse:
        """Check for hallucinations."""
        content = response.content if isinstance(response, LLMResponse) else response

        warnings = []
        confidence = 1.0

        if not self._context:
            # No context to compare against
            return ProcessedResponse(
                content=content,
                original=content,
                confidence=confidence,
                warnings=["No context provided for hallucination check"],
                metadata={"hallucination_check": "skipped"},
            )

        # Simple heuristic checks
        context_lower = self._context.lower()
        content_lower = content.lower()

        # Extract potential claims (sentences with specific patterns)
        # This is a simplified check - production would use more sophisticated methods
        sentences = content.split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for specific claim patterns
            claim_patterns = [
                r'\b(is|was|are|were)\s+\w+',  # "X is Y"
                r'\b(has|have|had)\s+\w+',     # "X has Y"
                r'\b\d+\s*(percent|%)',         # Percentages
                r'\b(always|never|all|none)\b', # Absolutes
            ]

            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Check if key terms appear in context
                    words = sentence.lower().split()
                    significant_words = [w for w in words if len(w) > 4]

                    in_context = sum(1 for w in significant_words if w in context_lower)
                    coverage = in_context / len(significant_words) if significant_words else 0

                    if coverage < 0.3:
                        if self._strict:
                            warnings.append(f"Potential unsupported claim: '{sentence[:50]}...'")
                            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))

        return ProcessedResponse(
            content=content,
            original=content,
            confidence=confidence,
            warnings=warnings,
            metadata={
                "hallucination_check": "completed",
                "potential_issues": len(warnings),
            },
        )
