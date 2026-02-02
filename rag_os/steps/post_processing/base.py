"""Base post-processing step for RAG OS."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.steps.llm.base import LLMResponse


@dataclass
class ProcessedResponse:
    """Processed LLM response.

    Attributes:
        content: Processed content
        original: Original content
        citations: Extracted citations
        confidence: Response confidence score
        warnings: Any warnings about the response
        metadata: Additional metadata
    """
    content: str
    original: str = ""
    citations: list[dict[str, Any]] = None
    confidence: float = 1.0
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "original": self.original,
            "citations": self.citations,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class BasePostProcessingStep(Step):
    """Abstract base class for post-processing steps.

    Post-processing steps transform, validate, or enrich LLM responses.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.POST_PROCESSING,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)

    @property
    def input_schema(self) -> dict[str, Any]:
        """Input is LLMResponse or string."""
        return {
            "anyOf": [
                {"type": "string"},
                {"type": "object", "description": "LLMResponse"},
            ]
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Output is ProcessedResponse."""
        return {"type": "object", "description": "ProcessedResponse"}

    @abstractmethod
    def process(self, response: str | LLMResponse) -> ProcessedResponse:
        """Process the LLM response.

        Args:
            response: LLM response to process

        Returns:
            Processed response
        """
        pass

    def execute(self, context: StepContext) -> StepResult:
        """Execute the post-processing step."""
        import time
        start = time.time()

        data = context.data

        # Parse input
        if isinstance(data, str):
            response = data
        elif isinstance(data, LLMResponse):
            response = data
        elif isinstance(data, dict):
            if "content" in data:
                response = LLMResponse(
                    content=data["content"],
                    model=data.get("model", ""),
                    prompt_tokens=data.get("prompt_tokens", 0),
                    completion_tokens=data.get("completion_tokens", 0),
                    total_tokens=data.get("total_tokens", 0),
                )
            else:
                response = str(data)
        else:
            return StepResult.fail(
                f"Invalid input type: {type(data)}",
                step_id=self.step_id,
            )

        try:
            processed = self.process(response)
            latency = (time.time() - start) * 1000

            return StepResult.ok(
                output=processed,
                confidence=processed.confidence,
                warning_count=len(processed.warnings),
            ).with_latency(latency)

        except Exception as e:
            return StepResult.fail(str(e), step_id=self.step_id)
