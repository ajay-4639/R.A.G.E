"""Document model for RAG OS."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
import hashlib
import json


class SourceType(str, Enum):
    """Type of document source."""

    # File types
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"

    # Web/API types
    URL = "url"
    API = "api"
    WEBPAGE = "webpage"

    # Storage types
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"

    # Other
    DATABASE = "database"
    EMAIL = "email"
    UNKNOWN = "unknown"


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class AccessControl:
    """
    Access control settings for a document.

    Defines who can access and use the document in retrieval.
    """

    owner_id: str | None = None
    team_ids: list[str] = field(default_factory=list)
    role_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    public: bool = False

    def can_access(
        self,
        user_id: str | None = None,
        user_teams: list[str] | None = None,
        user_roles: list[str] | None = None,
    ) -> bool:
        """
        Check if a user can access this document.

        Args:
            user_id: The user's ID
            user_teams: Teams the user belongs to
            user_roles: Roles the user has

        Returns:
            True if user can access, False otherwise
        """
        # Public documents are accessible to all
        if self.public:
            return True

        # Owner always has access
        if user_id and self.owner_id == user_id:
            return True

        # Check team membership
        if user_teams and self.team_ids:
            if any(team in self.team_ids for team in user_teams):
                return True

        # Check role membership
        if user_roles and self.role_ids:
            if any(role in self.role_ids for role in user_roles):
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "owner_id": self.owner_id,
            "team_ids": self.team_ids,
            "role_ids": self.role_ids,
            "tags": self.tags,
            "public": self.public,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessControl":
        """Create from dictionary."""
        return cls(
            owner_id=data.get("owner_id"),
            team_ids=data.get("team_ids", []),
            role_ids=data.get("role_ids", []),
            tags=data.get("tags", []),
            public=data.get("public", False),
        )


@dataclass
class Document:
    """
    A document in the RAG system.

    Represents a single document that can be chunked, embedded,
    and retrieved for RAG pipelines.

    Attributes:
        doc_id: Unique identifier for this document
        content: The text content of the document
        source_type: Type of the source (PDF, URL, etc.)
        source_uri: URI or path to the original source
        metadata: Additional metadata about the document
        acl: Access control settings
        created_at: When the document was created/ingested
        updated_at: When the document was last updated
        expires_at: Optional expiration time (for temporal validity)
        content_hash: Hash of content for deduplication
        title: Optional document title
        language: Optional language code (e.g., "en", "es")
    """

    content: str
    source_type: SourceType = SourceType.UNKNOWN
    source_uri: str = ""
    doc_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    acl: AccessControl = field(default_factory=AccessControl)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime | None = None
    content_hash: str = ""
    title: str = ""
    language: str = ""

    def __post_init__(self) -> None:
        """Compute content hash if not provided."""
        if not self.content_hash and self.content:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def is_expired(self) -> bool:
        """Check if document has expired."""
        if self.expires_at is None:
            return False
        return _utc_now() > self.expires_at

    @property
    def content_length(self) -> int:
        """Get content length in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Estimate word count."""
        return len(self.content.split())

    def update_content(self, new_content: str) -> None:
        """Update document content and recalculate hash."""
        self.content = new_content
        self.content_hash = self._compute_hash()
        self.updated_at = _utc_now()

    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata fields."""
        self.metadata.update(kwargs)
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source_type": self.source_type.value,
            "source_uri": self.source_uri,
            "metadata": self.metadata,
            "acl": self.acl.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "content_hash": self.content_hash,
            "title": self.title,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(
            doc_id=data.get("doc_id", str(uuid4())),
            content=data["content"],
            source_type=SourceType(data.get("source_type", "unknown")),
            source_uri=data.get("source_uri", ""),
            metadata=data.get("metadata", {}),
            acl=AccessControl.from_dict(data.get("acl", {})),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else _utc_now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else _utc_now(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            content_hash=data.get("content_hash", ""),
            title=data.get("title", ""),
            language=data.get("language", ""),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"Document(doc_id='{self.doc_id[:8]}...', "
            f"source_type={self.source_type}, "
            f"content_length={self.content_length})"
        )


def create_document(
    content: str,
    source_type: SourceType | str = SourceType.UNKNOWN,
    source_uri: str = "",
    title: str = "",
    owner_id: str | None = None,
    public: bool = False,
    **metadata: Any,
) -> Document:
    """
    Factory function to create a document with common defaults.

    Args:
        content: Document content
        source_type: Type of source
        source_uri: URI to original source
        title: Document title
        owner_id: Owner user ID
        public: Whether document is public
        **metadata: Additional metadata

    Returns:
        New Document instance
    """
    if isinstance(source_type, str):
        source_type = SourceType(source_type)

    acl = AccessControl(owner_id=owner_id, public=public)

    return Document(
        content=content,
        source_type=source_type,
        source_uri=source_uri,
        title=title,
        acl=acl,
        metadata=metadata,
    )
