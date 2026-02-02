"""Prompt assembly implementations for RAG OS."""

from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.index import SearchResult
from rag_os.steps.prompt_assembly.base import BasePromptAssemblyStep, AssembledPrompt


@register_step(
    name="SimplePromptAssemblyStep",
    step_type=StepType.PROMPT_ASSEMBLY,
    description="Simple prompt assembly with context",
    version="1.0.0",
)
class SimplePromptAssemblyStep(BasePromptAssemblyStep):
    """Simple prompt assembly that concatenates context."""

    def assemble(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AssembledPrompt:
        """Assemble a simple prompt with context."""
        config = self.get_config()

        # Build context from results
        context_parts = []
        for i, result in enumerate(results):
            if config.cite_sources:
                context_parts.append(f"[{i+1}] {result.content}")
            else:
                context_parts.append(result.content)

        context = config.separator.join(context_parts)

        # Build prompt
        if context:
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        return AssembledPrompt(
            prompt=prompt,
            system=config.system_prompt,
            user=query,
            context=context,
            metadata={"result_count": len(results)},
        )


@register_step(
    name="TemplatePromptAssemblyStep",
    step_type=StepType.PROMPT_ASSEMBLY,
    description="Template-based prompt assembly",
    version="1.0.0",
)
class TemplatePromptAssemblyStep(BasePromptAssemblyStep):
    """Template-based prompt assembly using string templates."""

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._template = config.get("template") if config else None

    def get_template(self) -> str:
        """Get the prompt template."""
        return self._template or """You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}

Instructions: {instructions}

Answer:"""

    def assemble(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AssembledPrompt:
        """Assemble prompt using template."""
        config = self.get_config()

        # Build context
        context_parts = []
        for i, result in enumerate(results):
            if config.cite_sources:
                context_parts.append(f"[Source {i+1}]: {result.content}")
            else:
                context_parts.append(result.content)

        context = config.separator.join(context_parts)

        # Get custom instructions
        instructions = self.config.get("instructions", "Provide a helpful and accurate response.") if self.config else "Provide a helpful and accurate response."

        # Format template
        template = self.get_template()
        prompt = template.format(
            context=context or "No context available.",
            query=query,
            instructions=instructions,
            system=config.system_prompt,
        )

        return AssembledPrompt(
            prompt=prompt,
            system=config.system_prompt,
            user=query,
            context=context,
            metadata={
                "result_count": len(results),
                "template_used": True,
            },
        )


@register_step(
    name="ChatPromptAssemblyStep",
    step_type=StepType.PROMPT_ASSEMBLY,
    description="Chat message format prompt assembly",
    version="1.0.0",
)
class ChatPromptAssemblyStep(BasePromptAssemblyStep):
    """Prompt assembly for chat/message-based LLM APIs."""

    def assemble(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AssembledPrompt:
        """Assemble prompt as chat messages."""
        config = self.get_config()

        # Build context
        context_parts = []
        for i, result in enumerate(results):
            if config.cite_sources:
                context_parts.append(f"[{i+1}] {result.content}")
            else:
                context_parts.append(result.content)

        context = config.separator.join(context_parts)

        # Build messages
        messages = [
            {"role": "system", "content": config.system_prompt},
        ]

        # Add context as assistant message or user context
        if context:
            context_message = f"Here is relevant context to help answer the question:\n\n{context}"
            messages.append({"role": "user", "content": context_message})
            messages.append({"role": "assistant", "content": "I understand. I'll use this context to answer your question."})

        # Add user query
        messages.append({"role": "user", "content": query})

        # Build full prompt for logging/debugging
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        return AssembledPrompt(
            prompt=prompt,
            messages=messages,
            system=config.system_prompt,
            user=query,
            context=context,
            metadata={
                "result_count": len(results),
                "message_count": len(messages),
            },
        )


@register_step(
    name="RAGPromptAssemblyStep",
    step_type=StepType.PROMPT_ASSEMBLY,
    description="RAG-optimized prompt assembly",
    version="1.0.0",
)
class RAGPromptAssemblyStep(BasePromptAssemblyStep):
    """RAG-optimized prompt assembly with structured output instructions."""

    def assemble(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AssembledPrompt:
        """Assemble RAG-optimized prompt."""
        config = self.get_config()

        # Build context with source citations
        context_parts = []
        sources = []
        for i, result in enumerate(results):
            source_id = f"[{i+1}]"
            context_parts.append(f"{source_id} {result.content}")
            sources.append({
                "id": source_id,
                "doc_id": result.doc_id,
                "chunk_id": result.chunk_id,
            })

        context = config.separator.join(context_parts)

        # Build system prompt for RAG
        system = f"""{config.system_prompt}

Instructions:
- Use the provided context to answer the question
- If the context doesn't contain relevant information, say so
- Cite sources using [1], [2], etc. when referring to specific context
- Be concise and accurate"""

        # Build user message
        if context:
            user_message = f"""Context:
{context}

Question: {query}"""
        else:
            user_message = f"Question: {query}\n\n(No context available)"

        # Build messages for chat models
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        prompt = f"{system}\n\n{user_message}"

        return AssembledPrompt(
            prompt=prompt,
            messages=messages,
            system=system,
            user=query,
            context=context,
            metadata={
                "result_count": len(results),
                "sources": sources,
            },
        )
