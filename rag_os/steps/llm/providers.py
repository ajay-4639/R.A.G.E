"""LLM provider implementations for RAG OS."""

from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.steps.prompt_assembly.base import AssembledPrompt
from rag_os.steps.llm.base import BaseLLMStep, LLMResponse


@register_step(
    name="MockLLMStep",
    step_type=StepType.LLM_EXECUTION,
    description="Mock LLM for testing",
    version="1.0.0",
)
class MockLLMStep(BaseLLMStep):
    """Mock LLM step for testing without API calls."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._response_template = config.get("response_template") if config else None

    def generate(self, prompt: str | AssembledPrompt) -> LLMResponse:
        """Generate a mock response."""
        if isinstance(prompt, AssembledPrompt):
            prompt_text = prompt.prompt
            query = prompt.user
        else:
            prompt_text = prompt
            query = prompt

        # Generate mock response
        if self._response_template:
            content = self._response_template.format(query=query)
        else:
            content = f"This is a mock response to the query: {query}"

        # Estimate tokens (rough approximation)
        prompt_tokens = len(prompt_text.split()) * 1.3
        completion_tokens = len(content.split()) * 1.3

        return LLMResponse(
            content=content,
            model="mock-model",
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens),
            finish_reason="stop",
        )


@register_step(
    name="OpenAILLMStep",
    step_type=StepType.LLM_EXECUTION,
    description="OpenAI LLM integration",
    version="1.0.0",
)
class OpenAILLMStep(BaseLLMStep):
    """LLM step using OpenAI's API."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )

            config = self.get_config()
            kwargs: dict[str, Any] = {}
            if config.api_key:
                kwargs["api_key"] = config.api_key

            self._client = OpenAI(**kwargs)

        return self._client

    def generate(self, prompt: str | AssembledPrompt) -> LLMResponse:
        """Generate response using OpenAI API."""
        client = self._get_client()
        config = self.get_config()

        # Build messages
        if isinstance(prompt, AssembledPrompt) and prompt.messages:
            messages = prompt.messages
        elif isinstance(prompt, AssembledPrompt):
            messages = [
                {"role": "system", "content": prompt.system or "You are a helpful assistant."},
                {"role": "user", "content": prompt.prompt or prompt.user},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        # Call API
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop if config.stop else None,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )


@register_step(
    name="AnthropicLLMStep",
    step_type=StepType.LLM_EXECUTION,
    description="Anthropic Claude LLM integration",
    version="1.0.0",
)
class AnthropicLLMStep(BaseLLMStep):
    """LLM step using Anthropic's Claude API."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )

            config = self.get_config()
            kwargs: dict[str, Any] = {}
            if config.api_key:
                kwargs["api_key"] = config.api_key

            self._client = anthropic.Anthropic(**kwargs)

        return self._client

    def generate(self, prompt: str | AssembledPrompt) -> LLMResponse:
        """Generate response using Anthropic API."""
        client = self._get_client()
        config = self.get_config()

        # Build messages (Anthropic format)
        if isinstance(prompt, AssembledPrompt):
            system = prompt.system or "You are a helpful assistant."
            # Convert messages to Anthropic format
            if prompt.messages:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in prompt.messages
                    if m["role"] != "system"
                ]
            else:
                messages = [{"role": "user", "content": prompt.prompt or prompt.user}]
        else:
            system = "You are a helpful assistant."
            messages = [{"role": "user", "content": prompt}]

        # Call API
        response = client.messages.create(
            model=config.model if "claude" in config.model else "claude-3-haiku-20240307",
            max_tokens=config.max_tokens,
            system=system,
            messages=messages,
        )

        content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "stop",
            raw_response=response,
        )
