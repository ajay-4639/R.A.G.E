"""Base LLM execution step for RAG OS."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.steps.prompt_assembly.base import AssembledPrompt


@dataclass
class LLMConfig:
    """Configuration for LLM execution.

    Attributes:
        model: Model name/ID
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        top_p: Top-p sampling parameter
        stop: Stop sequences
        api_key: API key (optional)
        timeout: Request timeout in seconds
    """
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    api_key: str | None = None
    timeout: float = 60.0


@dataclass
class LLMResponse:
    """Response from LLM execution.

    Attributes:
        content: Generated text content
        model: Model used
        prompt_tokens: Input token count
        completion_tokens: Output token count
        total_tokens: Total token count
        finish_reason: Why generation stopped
        raw_response: Raw API response
    """
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    raw_response: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
        }


class BaseLLMStep(Step):
    """Abstract base class for LLM execution steps.

    LLM steps take an assembled prompt and generate a response.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.LLM_EXECUTION,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)

    def get_config(self) -> LLMConfig:
        """Get typed configuration."""
        cfg = self.config or {}
        return LLMConfig(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 1000),
            top_p=cfg.get("top_p", 1.0),
            stop=cfg.get("stop", []),
            api_key=cfg.get("api_key"),
            timeout=cfg.get("timeout", 60.0),
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """Input is an AssembledPrompt or prompt string."""
        return {
            "anyOf": [
                {"type": "string"},
                {"type": "object", "description": "AssembledPrompt"},
            ]
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Output is LLMResponse."""
        return {"type": "object", "description": "LLMResponse"}

    @abstractmethod
    def generate(
        self,
        prompt: str | AssembledPrompt,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt string or AssembledPrompt

        Returns:
            LLM response with generated content
        """
        pass

    def execute(self, context: StepContext) -> StepResult:
        """Execute the LLM step."""
        import time
        start = time.time()

        data = context.data

        # Parse input
        if isinstance(data, str):
            prompt = data
        elif isinstance(data, AssembledPrompt):
            prompt = data
        elif isinstance(data, dict):
            # Reconstruct AssembledPrompt
            prompt = AssembledPrompt(
                prompt=data.get("prompt", ""),
                messages=data.get("messages", []),
                system=data.get("system", ""),
                user=data.get("user", ""),
                context=data.get("context", ""),
                metadata=data.get("metadata", {}),
            )
        else:
            return StepResult.fail(
                f"Invalid input type: {type(data)}",
                step_id=self.step_id,
            )

        try:
            response = self.generate(prompt)
            latency = (time.time() - start) * 1000

            return StepResult.ok(
                output=response,
                model=response.model,
                finish_reason=response.finish_reason,
            ).with_latency(latency).with_token_usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )

        except Exception as e:
            return StepResult.fail(str(e), step_id=self.step_id)
