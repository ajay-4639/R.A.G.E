"""Unit tests for prompt assembly, LLM, and post-processing steps."""

import pytest

from rag_os.core.context import StepContext
from rag_os.models.index import SearchResult
from rag_os.steps.prompt_assembly import (
    SimplePromptAssemblyStep,
    TemplatePromptAssemblyStep,
    ChatPromptAssemblyStep,
)
from rag_os.steps.prompt_assembly.base import AssembledPrompt
from rag_os.steps.llm import MockLLMStep, LLMResponse
from rag_os.steps.post_processing import (
    CitationExtractionStep,
    ResponseFormattingStep,
    AnswerValidationStep,
    HallucinationCheckStep,
)


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results."""
    return [
        SearchResult(
            chunk_id="c1", doc_id="d1",
            content="Python is a versatile programming language.",
            score=0.9,
        ),
        SearchResult(
            chunk_id="c2", doc_id="d1",
            content="Machine learning enables pattern recognition.",
            score=0.8,
        ),
        SearchResult(
            chunk_id="c3", doc_id="d2",
            content="Data science combines statistics and coding.",
            score=0.7,
        ),
    ]


class TestSimplePromptAssemblyStep:
    """Tests for SimplePromptAssemblyStep."""

    def test_assemble_basic(self, sample_results):
        """Basic prompt assembly works."""
        step = SimplePromptAssemblyStep(step_id="assemble", config={})

        prompt = step.assemble("What is Python?", sample_results)

        assert isinstance(prompt, AssembledPrompt)
        assert "Python" in prompt.context
        assert "What is Python?" in prompt.prompt

    def test_citations_included(self, sample_results):
        """Citations are included when configured."""
        step = SimplePromptAssemblyStep(
            step_id="assemble",
            config={"cite_sources": True},
        )

        prompt = step.assemble("test query", sample_results)

        assert "[1]" in prompt.context
        assert "[2]" in prompt.context

    def test_execute(self, sample_results):
        """Execute method works."""
        step = SimplePromptAssemblyStep(step_id="assemble", config={})

        context = StepContext(data={
            "query": "What is Python?",
            "results": sample_results,
        })
        result = step.execute(context)

        assert result.success
        assert isinstance(result.output, AssembledPrompt)

    def test_empty_results(self):
        """Handles empty results."""
        step = SimplePromptAssemblyStep(step_id="assemble", config={})

        prompt = step.assemble("test query", [])

        assert "test query" in prompt.prompt


class TestTemplatePromptAssemblyStep:
    """Tests for TemplatePromptAssemblyStep."""

    def test_custom_template(self, sample_results):
        """Custom template is used."""
        step = TemplatePromptAssemblyStep(
            step_id="assemble",
            config={
                "template": "Context: {context}\nQ: {query}\nInstructions: {instructions}",
                "instructions": "Be concise.",
            },
        )

        prompt = step.assemble("test query", sample_results)

        assert "Be concise" in prompt.prompt
        assert "Q: test query" in prompt.prompt


class TestChatPromptAssemblyStep:
    """Tests for ChatPromptAssemblyStep."""

    def test_chat_messages_format(self, sample_results):
        """Chat messages are properly formatted."""
        step = ChatPromptAssemblyStep(
            step_id="assemble",
            config={"system_prompt": "You are a helpful AI."},
        )

        prompt = step.assemble("What is ML?", sample_results)

        assert len(prompt.messages) > 0
        assert prompt.messages[0]["role"] == "system"
        assert "helpful" in prompt.messages[0]["content"]

    def test_user_message_included(self, sample_results):
        """User query is in messages."""
        step = ChatPromptAssemblyStep(step_id="assemble", config={})

        prompt = step.assemble("test query", sample_results)

        user_messages = [m for m in prompt.messages if m["role"] == "user"]
        assert any("test query" in m["content"] for m in user_messages)


class TestMockLLMStep:
    """Tests for MockLLMStep."""

    def test_generate_basic(self):
        """Basic generation works."""
        step = MockLLMStep(step_id="llm", config={})

        response = step.generate("What is 2+2?")

        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.model == "mock-model"

    def test_generate_with_template(self):
        """Response template is used."""
        step = MockLLMStep(
            step_id="llm",
            config={"response_template": "Answer to '{query}' is: ..."},
        )

        response = step.generate("test")

        assert "Answer to" in response.content

    def test_generate_from_assembled_prompt(self, sample_results):
        """Can generate from AssembledPrompt."""
        prompt = AssembledPrompt(
            prompt="Test prompt",
            user="test query",
            context="some context",
        )

        step = MockLLMStep(step_id="llm", config={})
        response = step.generate(prompt)

        assert response.content

    def test_execute(self):
        """Execute method works."""
        step = MockLLMStep(step_id="llm", config={})

        context = StepContext(data="What is AI?")
        result = step.execute(context)

        assert result.success
        assert isinstance(result.output, LLMResponse)

    def test_token_usage(self):
        """Token usage is estimated."""
        step = MockLLMStep(step_id="llm", config={})

        response = step.generate("Short query")

        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens


class TestCitationExtractionStep:
    """Tests for CitationExtractionStep."""

    def test_extract_citations(self):
        """Citations are extracted."""
        step = CitationExtractionStep(step_id="cite", config={})

        response = LLMResponse(
            content="According to [1], Python is great. As noted in [2], it's popular.",
            model="test",
        )
        processed = step.process(response)

        assert len(processed.citations) == 2

    def test_citation_indices(self):
        """Citation indices are correct."""
        step = CitationExtractionStep(step_id="cite", config={})

        response = "See [1] and [3] for details."
        processed = step.process(response)

        indices = [c["index"] for c in processed.citations]
        assert 0 in indices  # [1] -> index 0
        assert 2 in indices  # [3] -> index 2

    def test_no_citations(self):
        """Handles response without citations."""
        step = CitationExtractionStep(step_id="cite", config={})

        processed = step.process("No citations here.")

        assert processed.citations == []


class TestResponseFormattingStep:
    """Tests for ResponseFormattingStep."""

    def test_strip_whitespace(self):
        """Whitespace is stripped."""
        step = ResponseFormattingStep(step_id="format", config={})

        processed = step.process("  Answer here.  \n\n")

        assert processed.content == "Answer here."

    def test_remove_thinking(self):
        """Thinking blocks are removed."""
        step = ResponseFormattingStep(step_id="format", config={})

        response = "Let me think about this. The answer is 42."
        processed = step.process(response)

        assert "Let me think" not in processed.content

    def test_max_length(self):
        """Response is truncated if too long."""
        step = ResponseFormattingStep(
            step_id="format",
            config={"max_length": 20},
        )

        processed = step.process("This is a very long response that should be truncated.")

        assert len(processed.content) <= 23  # 20 + "..."


class TestAnswerValidationStep:
    """Tests for AnswerValidationStep."""

    def test_short_response_warning(self):
        """Short responses get warnings."""
        step = AnswerValidationStep(
            step_id="validate",
            config={"min_length": 50},
        )

        processed = step.process("Short.")

        assert len(processed.warnings) > 0
        assert processed.confidence < 1.0

    def test_uncertainty_detection(self):
        """Uncertainty phrases are detected."""
        step = AnswerValidationStep(step_id="validate", config={})

        processed = step.process("I don't know the answer to that question.")

        assert any("uncertainty" in w.lower() for w in processed.warnings)

    def test_good_response(self):
        """Good responses pass validation."""
        step = AnswerValidationStep(step_id="validate", config={})

        processed = step.process("Python is a programming language used for various applications including web development and data science.")

        assert processed.confidence > 0.8


class TestHallucinationCheckStep:
    """Tests for HallucinationCheckStep."""

    def test_with_context(self):
        """Checks against provided context."""
        step = HallucinationCheckStep(
            step_id="hallucination",
            config={"context": "Python is a programming language."},
        )

        processed = step.process("Python is a programming language used widely.")

        assert processed.metadata["hallucination_check"] == "completed"

    def test_no_context(self):
        """Handles missing context."""
        step = HallucinationCheckStep(step_id="hallucination", config={})

        processed = step.process("Some response.")

        assert "skipped" in processed.metadata["hallucination_check"]


class TestIntegration:
    """Integration tests for the full pipeline flow."""

    def test_full_rag_pipeline(self, sample_results):
        """Test a complete RAG pipeline flow."""
        # 1. Assemble prompt
        assembler = SimplePromptAssemblyStep(step_id="assemble", config={})
        prompt = assembler.assemble("What is Python?", sample_results)

        # 2. Generate response
        llm = MockLLMStep(step_id="llm", config={})
        response = llm.generate(prompt)

        # 3. Post-process
        formatter = ResponseFormattingStep(step_id="format", config={})
        formatted = formatter.process(response)

        validator = AnswerValidationStep(step_id="validate", config={})
        validated = validator.process(formatted.content)

        assert validated.content
        assert validated.confidence > 0

    def test_execute_chain(self, sample_results):
        """Test execute methods in chain."""
        assembler = SimplePromptAssemblyStep(step_id="assemble", config={})
        llm = MockLLMStep(step_id="llm", config={})

        # Assemble
        ctx1 = StepContext(data={"query": "test", "results": sample_results})
        result1 = assembler.execute(ctx1)
        assert result1.success

        # Generate
        ctx2 = StepContext(data=result1.output)
        result2 = llm.execute(ctx2)
        assert result2.success
        assert isinstance(result2.output, LLMResponse)
