"""RAG Session - Conversational interface for RAG pipelines."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
        timestamp: When the message was created
        metadata: Additional metadata
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SessionConfig:
    """Configuration for RAG session.

    Attributes:
        max_history: Maximum messages to keep in history
        system_prompt: System prompt for the session
        include_context: Whether to include retrieved context in history
    """
    max_history: int = 10
    system_prompt: str | None = None
    include_context: bool = False


class RAGSession:
    """A conversational session for RAG.

    Maintains conversation history and context for multi-turn
    interactions with a RAG pipeline.

    Usage:
        session = RAGSession(client)
        response = session.chat("What is Python?")
        response = session.chat("Tell me more about its history")
    """

    def __init__(
        self,
        client: Any,  # RAGClient
        pipeline: str = "default",
        config: SessionConfig | None = None,
    ):
        """Initialize session.

        Args:
            client: RAGClient instance
            pipeline: Pipeline to use
            config: Session configuration
        """
        self._client = client
        self._pipeline = pipeline
        self._config = config or SessionConfig()
        self._session_id = str(uuid4())[:8]
        self._history: list[Message] = []
        self._context: dict[str, Any] = {}
        self._created_at = _utc_now()

        # Add system prompt if configured
        if self._config.system_prompt:
            self._history.append(Message(
                role="system",
                content=self._config.system_prompt,
            ))

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def history(self) -> list[Message]:
        """Get conversation history."""
        return self._history.copy()

    @property
    def message_count(self) -> int:
        """Get number of messages."""
        return len(self._history)

    def chat(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Send a message and get a response.

        Args:
            message: User message
            context: Additional context

        Returns:
            Assistant response
        """
        # Add user message to history
        user_msg = Message(role="user", content=message)
        self._history.append(user_msg)

        # Build context with history
        query_context = {
            "session_id": self._session_id,
            "history": self._format_history(),
        }
        if context:
            query_context.update(context)

        # Query the pipeline
        result = self._client.query(
            query=message,
            pipeline=self._pipeline,
            context=query_context,
        )

        # Extract answer
        answer = result.get("answer", "")

        # Add assistant message to history
        assistant_msg = Message(
            role="assistant",
            content=answer,
            metadata={
                "latency_ms": result.get("latency_ms", 0),
                "pipeline": self._pipeline,
            },
        )
        self._history.append(assistant_msg)

        # Trim history if needed
        self._trim_history()

        return answer

    def _format_history(self) -> str:
        """Format history for context."""
        lines = []
        for msg in self._history:
            if msg.role == "system":
                continue
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def _trim_history(self) -> None:
        """Trim history to max size."""
        # Keep system messages
        system_msgs = [m for m in self._history if m.role == "system"]
        other_msgs = [m for m in self._history if m.role != "system"]

        # Keep last N non-system messages
        if len(other_msgs) > self._config.max_history:
            other_msgs = other_msgs[-self._config.max_history:]

        self._history = system_msgs + other_msgs

    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self._history = [m for m in self._history if m.role == "system"]

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key
            default: Default value

        Returns:
            Context value or default
        """
        return self._context.get(key, default)

    def get_last_response(self) -> Message | None:
        """Get the last assistant response."""
        for msg in reversed(self._history):
            if msg.role == "assistant":
                return msg
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self._session_id,
            "pipeline": self._pipeline,
            "created_at": self._created_at.isoformat(),
            "message_count": self.message_count,
            "history": [m.to_dict() for m in self._history],
            "context": self._context,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        client: Any,
    ) -> "RAGSession":
        """Create session from dictionary.

        Args:
            data: Session data
            client: RAGClient instance

        Returns:
            RAGSession instance
        """
        session = cls(client, data["pipeline"])
        session._session_id = data["session_id"]
        session._created_at = datetime.fromisoformat(data["created_at"])
        session._context = data.get("context", {})

        # Restore history
        session._history = []
        for msg_data in data.get("history", []):
            session._history.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data.get("metadata", {}),
            ))

        return session


class MultiSessionManager:
    """Manages multiple RAG sessions."""

    def __init__(self, client: Any):
        """Initialize manager.

        Args:
            client: RAGClient instance
        """
        self._client = client
        self._sessions: dict[str, RAGSession] = {}

    def create_session(
        self,
        pipeline: str = "default",
        config: SessionConfig | None = None,
    ) -> RAGSession:
        """Create a new session.

        Args:
            pipeline: Pipeline to use
            config: Session configuration

        Returns:
            New RAGSession
        """
        session = RAGSession(self._client, pipeline, config)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> RAGSession | None:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            RAGSession or None
        """
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """Get list of session IDs."""
        return list(self._sessions.keys())

    def clear_all(self) -> int:
        """Clear all sessions.

        Returns:
            Number of sessions cleared
        """
        count = len(self._sessions)
        self._sessions.clear()
        return count
