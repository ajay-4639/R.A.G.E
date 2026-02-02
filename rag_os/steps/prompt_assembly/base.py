"""Base prompt assembly step for RAG OS."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.models.index import SearchResult


@dataclass
class PromptConfig:
    """Configuration for prompt assembly.

    Attributes:
        system_prompt: System message for the LLM
        max_context_tokens: Maximum tokens for context
        include_metadata: Whether to include result metadata
        cite_sources: Whether to add source citations
        separator: Separator between context chunks
    """
    system_prompt: str = "You are a helpful assistant."
    max_context_tokens: int = 4000
    include_metadata: bool = False
    cite_sources: bool = True
    separator: str = "\n\n---\n\n"


@dataclass
class AssembledPrompt:
    """The assembled prompt ready for LLM.

    Attributes:
        prompt: The final prompt text (for completion models)
        messages: List of messages (for chat models)
        system: System message
        user: User query
        context: Retrieved context
        metadata: Additional metadata
    """
    prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    system: str = ""
    user: str = ""
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "system": self.system,
            "user": self.user,
            "context": self.context,
            "metadata": self.metadata,
        }


class BasePromptAssemblyStep(Step):
    """Abstract base class for prompt assembly steps.

    Prompt assembly combines the query, retrieved context, and
    system instructions into a prompt for the LLM.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.PROMPT_ASSEMBLY,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)

    def get_config(self) -> PromptConfig:
        """Get typed configuration."""
        cfg = self.config or {}
        return PromptConfig(
            system_prompt=cfg.get("system_prompt", "You are a helpful assistant."),
            max_context_tokens=cfg.get("max_context_tokens", 4000),
            include_metadata=cfg.get("include_metadata", False),
            cite_sources=cfg.get("cite_sources", True),
            separator=cfg.get("separator", "\n\n---\n\n"),
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """Input is query and search results."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {"type": "array"},
            },
            "required": ["query"],
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Output is assembled prompt."""
        return {
            "type": "object",
            "description": "AssembledPrompt",
        }

    @abstractmethod
    def assemble(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AssembledPrompt:
        """Assemble the prompt from query and results.

        Args:
            query: User query
            results: Retrieved search results

        Returns:
            Assembled prompt ready for LLM
        """
        pass

    def execute(self, context: StepContext) -> StepResult:
        """Execute the prompt assembly step."""
        import time
        start = time.time()

        data = context.data

        # Parse input
        if isinstance(data, dict):
            query = data.get("query", "")
            raw_results = data.get("results", [])
        elif isinstance(data, str):
            query = data
            raw_results = context.metadata.get("results", [])
        else:
            return StepResult.fail(
                f"Invalid input type: {type(data)}",
                step_id=self.step_id,
            )

        if not query:
            return StepResult.fail("Query is required", step_id=self.step_id)

        # Convert to SearchResult objects
        results: list[SearchResult] = []
        for r in raw_results:
            if isinstance(r, SearchResult):
                results.append(r)
            elif isinstance(r, dict):
                results.append(SearchResult.from_dict(r))

        try:
            assembled = self.assemble(query, results)
            latency = (time.time() - start) * 1000

            return StepResult.ok(
                output=assembled,
                context_chunks=len(results),
                prompt_length=len(assembled.prompt),
            ).with_latency(latency)

        except Exception as e:
            return StepResult.fail(str(e), step_id=self.step_id)
