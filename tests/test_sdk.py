"""Tests for RAG OS SDK."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from rag_os.sdk import (
    RAGClient,
    RAGClientConfig,
    PipelineBuilder,
    StepBuilder,
    RAGSession,
    SessionConfig,
)
from rag_os.sdk.session import Message, MultiSessionManager
from rag_os.core.spec import PipelineSpec, StepSpec
from rag_os.core.types import StepType


# =============================================================================
# StepBuilder Tests
# =============================================================================

class TestStepBuilder:
    """Tests for StepBuilder."""

    def test_basic_build(self):
        """Test basic step building."""
        step = (
            StepBuilder("step_1")
            .with_class("MockStep")
            .build()
        )

        assert step.step_id == "step_1"
        assert step.step_class == "MockStep"

    def test_with_config(self):
        """Test step with configuration."""
        step = (
            StepBuilder("step_1")
            .with_class("MockStep")
            .with_config(param1="value1", param2=123)
            .build()
        )

        assert step.config == {"param1": "value1", "param2": 123}

    def test_with_dependencies(self):
        """Test step with dependencies."""
        step = (
            StepBuilder("step_2")
            .with_class("MockStep")
            .depends_on("step_1", "step_0")
            .build()
        )

        assert step.dependencies == ["step_1", "step_0"]

    def test_with_fallback(self):
        """Test step with fallback."""
        step = (
            StepBuilder("step_1")
            .with_class("MockStep")
            .with_fallback("fallback_step")
            .build()
        )

        assert step.fallback_step == "fallback_step"

    def test_disabled_step(self):
        """Test disabled step."""
        step = (
            StepBuilder("step_1")
            .with_class("MockStep")
            .disabled()
            .build()
        )

        assert step.enabled is False

    def test_with_retry(self):
        """Test step with retry."""
        step = (
            StepBuilder("step_1")
            .with_class("MockStep")
            .with_retry(3)
            .build()
        )

        assert step.retry_policy is not None
        assert step.retry_policy["max_retries"] == 3

    def test_missing_class_raises(self):
        """Test that missing class raises error."""
        with pytest.raises(ValueError, match="Step class is required"):
            StepBuilder("step_1").build()


# =============================================================================
# PipelineBuilder Tests
# =============================================================================

class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_basic_build(self):
        """Test basic pipeline building."""
        pipeline = (
            PipelineBuilder("test_pipeline")
            .add_step(
                StepBuilder("step_1")
                .with_class("MockStep")
            )
            .build()
        )

        assert pipeline.name == "test_pipeline"
        assert len(pipeline.steps) == 1

    def test_with_version_and_description(self):
        """Test pipeline with version and description."""
        pipeline = (
            PipelineBuilder("test")
            .with_version("2.0.0")
            .with_description("Test pipeline")
            .add_step(StepBuilder("s1").with_class("Mock"))
            .build()
        )

        assert pipeline.version == "2.0.0"
        assert pipeline.description == "Test pipeline"

    def test_multiple_steps(self):
        """Test pipeline with multiple steps."""
        pipeline = (
            PipelineBuilder("test")
            .add_step(StepBuilder("s1").with_class("Step1"))
            .add_step(StepBuilder("s2").with_class("Step2").depends_on("s1"))
            .add_step(StepBuilder("s3").with_class("Step3").depends_on("s2"))
            .build()
        )

        assert len(pipeline.steps) == 3
        assert pipeline.steps[1].dependencies == ["s1"]

    def test_add_steps_batch(self):
        """Test adding multiple steps at once."""
        pipeline = (
            PipelineBuilder("test")
            .add_steps(
                StepBuilder("s1").with_class("Step1"),
                StepBuilder("s2").with_class("Step2"),
            )
            .build()
        )

        assert len(pipeline.steps) == 2

    def test_with_metadata(self):
        """Test pipeline with metadata."""
        pipeline = (
            PipelineBuilder("test")
            .with_metadata(author="test", team="qa")
            .add_step(StepBuilder("s1").with_class("Mock"))
            .build()
        )

        assert pipeline.metadata == {"author": "test", "team": "qa"}

    def test_convenience_methods(self):
        """Test convenience methods for common steps."""
        pipeline = (
            PipelineBuilder("rag_pipeline")
            .add_ingestion_step("ingest_1")
            .add_chunking_step("chunk_1", depends_on="ingest_1", chunk_size=500)
            .add_embedding_step("embed_1", depends_on="chunk_1")
            .add_retrieval_step("retrieve_1", top_k=5)
            .add_llm_step("llm_1", depends_on="retrieve_1")
            .build()
        )

        assert len(pipeline.steps) == 5
        assert pipeline.steps[0].step_class == "FileIngestionStep"
        assert pipeline.steps[1].step_class == "FixedSizeChunkingStep"
        assert pipeline.steps[1].config == {"chunk_size": 500}

    def test_empty_pipeline_raises(self):
        """Test that empty pipeline raises error."""
        with pytest.raises(ValueError, match="at least one step"):
            PipelineBuilder("empty").build()


# =============================================================================
# RAGClient Tests
# =============================================================================

class TestRAGClient:
    """Tests for RAGClient."""

    def test_create_client(self):
        """Test creating a client."""
        client = RAGClient()

        assert client.config is not None
        assert len(client.get_pipelines()) == 0

    def test_create_client_with_config(self):
        """Test creating client with config."""
        config = RAGClientConfig(
            enable_tracing=False,
            cache_enabled=True,
        )
        client = RAGClient(config)

        assert client.config.enable_tracing is False
        assert client.config.cache_enabled is True

    def test_load_pipeline_from_spec(self):
        """Test loading pipeline from spec."""
        client = RAGClient()

        spec = (
            PipelineBuilder("test")
            .add_step(StepBuilder("s1").with_class("Mock"))
            .build()
        )

        name = client.load_pipeline(spec)

        assert name == "test"
        assert "test" in client.get_pipelines()

    def test_load_pipeline_from_dict(self):
        """Test loading pipeline from dict."""
        client = RAGClient()

        spec_dict = {
            "name": "dict_pipeline",
            "version": "1.0.0",
            "steps": [
                {"step_id": "s1", "step_class": "Mock", "step_type": "llm_execution"}
            ]
        }

        name = client.load_pipeline(spec_dict)

        assert name == "dict_pipeline"

    def test_load_pipeline_with_name_override(self):
        """Test loading with name override."""
        client = RAGClient()

        spec = (
            PipelineBuilder("original")
            .add_step(StepBuilder("s1").with_class("Mock"))
            .build()
        )

        name = client.load_pipeline(spec, name="custom_name")

        assert name == "custom_name"
        assert "custom_name" in client.get_pipelines()

    def test_get_pipeline_info(self):
        """Test getting pipeline info."""
        client = RAGClient()

        spec = (
            PipelineBuilder("info_test")
            .with_description("Test description")
            .add_step(StepBuilder("s1").with_class("MockStep"))
            .build()
        )
        client.load_pipeline(spec)

        info = client.get_pipeline_info("info_test")

        assert info["name"] == "info_test"
        assert info["description"] == "Test description"
        assert len(info["steps"]) == 1

    def test_query_missing_pipeline_raises(self):
        """Test querying missing pipeline raises error."""
        client = RAGClient()

        with pytest.raises(ValueError, match="not loaded"):
            client.query("test query", pipeline="nonexistent")

    def test_register_hook(self):
        """Test registering hooks."""
        client = RAGClient()
        calls = []

        def hook(query, pipeline, context):
            calls.append(query)

        client.register_hook("pre_query", hook)

        # Hook should be registered
        assert len(client._hooks["pre_query"]) == 1

    def test_register_invalid_hook_raises(self):
        """Test registering invalid hook raises."""
        client = RAGClient()

        with pytest.raises(ValueError, match="Unknown event"):
            client.register_hook("invalid_event", lambda: None)

    def test_close(self):
        """Test closing client."""
        client = RAGClient()
        spec = PipelineBuilder("test").add_step(StepBuilder("s1").with_class("M")).build()
        client.load_pipeline(spec)

        client.close()

        assert len(client.get_pipelines()) == 0


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    """Tests for Message."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_message_to_dict(self):
        """Test serializing message."""
        msg = Message(role="assistant", content="Hi there")

        data = msg.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"
        assert "timestamp" in data


# =============================================================================
# RAGSession Tests
# =============================================================================

class TestRAGSession:
    """Tests for RAGSession."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock RAG client."""
        client = Mock()
        client.query.return_value = {
            "query": "test",
            "answer": "Test answer",
            "success": True,
            "latency_ms": 100,
        }
        return client

    def test_create_session(self, mock_client):
        """Test creating a session."""
        session = RAGSession(mock_client)

        assert session.session_id is not None
        assert session.message_count == 0

    def test_session_with_system_prompt(self, mock_client):
        """Test session with system prompt."""
        config = SessionConfig(system_prompt="You are a helpful assistant.")
        session = RAGSession(mock_client, config=config)

        # System message should be in history
        assert session.message_count == 1
        assert session.history[0].role == "system"

    def test_chat(self, mock_client):
        """Test basic chat."""
        session = RAGSession(mock_client, pipeline="qa")

        response = session.chat("Hello")

        assert response == "Test answer"
        assert session.message_count == 2  # user + assistant
        mock_client.query.assert_called_once()

    def test_chat_history(self, mock_client):
        """Test chat maintains history."""
        session = RAGSession(mock_client)

        session.chat("First message")
        session.chat("Second message")

        assert session.message_count == 4  # 2 user + 2 assistant

    def test_history_trimming(self, mock_client):
        """Test history is trimmed to max size."""
        config = SessionConfig(max_history=4)
        session = RAGSession(mock_client, config=config)

        # Send many messages
        for i in range(10):
            session.chat(f"Message {i}")

        # Should be trimmed (keeping last N pairs)
        assert session.message_count <= config.max_history

    def test_clear_history(self, mock_client):
        """Test clearing history."""
        config = SessionConfig(system_prompt="System")
        session = RAGSession(mock_client, config=config)
        session.chat("Hello")

        session.clear_history()

        # Should only have system prompt
        assert session.message_count == 1
        assert session.history[0].role == "system"

    def test_set_get_context(self, mock_client):
        """Test session context."""
        session = RAGSession(mock_client)

        session.set_context("user_id", "123")
        session.set_context("language", "en")

        assert session.get_context("user_id") == "123"
        assert session.get_context("language") == "en"
        assert session.get_context("missing", "default") == "default"

    def test_get_last_response(self, mock_client):
        """Test getting last response."""
        session = RAGSession(mock_client)

        session.chat("Hello")
        last = session.get_last_response()

        assert last is not None
        assert last.role == "assistant"
        assert last.content == "Test answer"

    def test_to_dict(self, mock_client):
        """Test serializing session."""
        session = RAGSession(mock_client, pipeline="test")
        session.chat("Hello")

        data = session.to_dict()

        assert data["session_id"] == session.session_id
        assert data["pipeline"] == "test"
        assert data["message_count"] == 2

    def test_from_dict(self, mock_client):
        """Test deserializing session."""
        original = RAGSession(mock_client, pipeline="test")
        original.chat("Hello")
        data = original.to_dict()

        restored = RAGSession.from_dict(data, mock_client)

        assert restored.session_id == original.session_id
        assert restored.message_count == original.message_count


# =============================================================================
# MultiSessionManager Tests
# =============================================================================

class TestMultiSessionManager:
    """Tests for MultiSessionManager."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        client = Mock()
        client.query.return_value = {
            "answer": "Test",
            "success": True,
            "latency_ms": 50,
        }
        return client

    def test_create_session(self, mock_client):
        """Test creating sessions."""
        manager = MultiSessionManager(mock_client)

        session1 = manager.create_session(pipeline="qa")
        session2 = manager.create_session(pipeline="chat")

        assert len(manager.list_sessions()) == 2
        assert session1.session_id != session2.session_id

    def test_get_session(self, mock_client):
        """Test getting a session."""
        manager = MultiSessionManager(mock_client)
        session = manager.create_session()

        retrieved = manager.get_session(session.session_id)

        assert retrieved is session

    def test_get_nonexistent_session(self, mock_client):
        """Test getting nonexistent session."""
        manager = MultiSessionManager(mock_client)

        result = manager.get_session("fake_id")

        assert result is None

    def test_delete_session(self, mock_client):
        """Test deleting a session."""
        manager = MultiSessionManager(mock_client)
        session = manager.create_session()

        result = manager.delete_session(session.session_id)

        assert result is True
        assert len(manager.list_sessions()) == 0

    def test_clear_all(self, mock_client):
        """Test clearing all sessions."""
        manager = MultiSessionManager(mock_client)
        manager.create_session()
        manager.create_session()
        manager.create_session()

        count = manager.clear_all()

        assert count == 3
        assert len(manager.list_sessions()) == 0
