"""Unit tests for document model."""

import pytest
import json
from datetime import datetime, timezone, timedelta

from rag_os.models.document import (
    Document,
    SourceType,
    AccessControl,
    create_document,
)


class TestSourceType:
    """Tests for SourceType enum."""

    def test_all_types_exist(self):
        """All expected source types exist."""
        types = [t.value for t in SourceType]
        assert "pdf" in types
        assert "url" in types
        assert "text" in types
        assert "markdown" in types


class TestAccessControl:
    """Tests for AccessControl."""

    def test_default_acl(self):
        """Default ACL is restrictive."""
        acl = AccessControl()
        assert acl.owner_id is None
        assert acl.team_ids == []
        assert not acl.public

    def test_public_access(self):
        """Public documents are accessible to all."""
        acl = AccessControl(public=True)

        assert acl.can_access()
        assert acl.can_access(user_id="anyone")

    def test_owner_access(self):
        """Owner can access their documents."""
        acl = AccessControl(owner_id="user123", public=False)

        assert acl.can_access(user_id="user123")
        assert not acl.can_access(user_id="other_user")

    def test_team_access(self):
        """Team members can access team documents."""
        acl = AccessControl(team_ids=["team_a", "team_b"])

        assert acl.can_access(user_teams=["team_a"])
        assert acl.can_access(user_teams=["team_b", "team_c"])
        assert not acl.can_access(user_teams=["team_c"])

    def test_role_access(self):
        """Users with matching roles can access."""
        acl = AccessControl(role_ids=["admin", "editor"])

        assert acl.can_access(user_roles=["admin"])
        assert not acl.can_access(user_roles=["viewer"])

    def test_no_access(self):
        """Private documents require proper credentials."""
        acl = AccessControl(owner_id="user123", public=False)

        assert not acl.can_access()
        assert not acl.can_access(user_id="wrong_user")

    def test_to_from_dict(self):
        """ACL can be serialized and deserialized."""
        acl = AccessControl(
            owner_id="user1",
            team_ids=["t1", "t2"],
            role_ids=["r1"],
            tags=["tag1"],
            public=False,
        )

        data = acl.to_dict()
        restored = AccessControl.from_dict(data)

        assert restored.owner_id == "user1"
        assert restored.team_ids == ["t1", "t2"]
        assert restored.role_ids == ["r1"]
        assert restored.tags == ["tag1"]


class TestDocument:
    """Tests for Document."""

    def test_basic_creation(self):
        """Can create a basic document."""
        doc = Document(content="Hello, world!")

        assert doc.content == "Hello, world!"
        assert doc.doc_id is not None
        assert len(doc.doc_id) == 36  # UUID format
        assert doc.content_hash != ""

    def test_with_all_fields(self):
        """Can create document with all fields."""
        doc = Document(
            content="Test content",
            source_type=SourceType.PDF,
            source_uri="/path/to/file.pdf",
            title="Test Document",
            metadata={"author": "Test"},
            language="en",
        )

        assert doc.source_type == SourceType.PDF
        assert doc.source_uri == "/path/to/file.pdf"
        assert doc.title == "Test Document"
        assert doc.metadata["author"] == "Test"
        assert doc.language == "en"

    def test_content_hash_computed(self):
        """Content hash is computed automatically."""
        doc1 = Document(content="same content")
        doc2 = Document(content="same content")
        doc3 = Document(content="different content")

        assert doc1.content_hash == doc2.content_hash
        assert doc1.content_hash != doc3.content_hash

    def test_content_length(self):
        """Content length property works."""
        doc = Document(content="Hello")
        assert doc.content_length == 5

    def test_word_count(self):
        """Word count estimation works."""
        doc = Document(content="Hello world this is a test")
        assert doc.word_count == 6

    def test_is_expired(self):
        """Expiration check works."""
        # Not expired
        future = datetime.now(timezone.utc) + timedelta(days=1)
        doc1 = Document(content="test", expires_at=future)
        assert not doc1.is_expired

        # Expired
        past = datetime.now(timezone.utc) - timedelta(days=1)
        doc2 = Document(content="test", expires_at=past)
        assert doc2.is_expired

        # No expiration
        doc3 = Document(content="test")
        assert not doc3.is_expired

    def test_update_content(self):
        """Content update works."""
        doc = Document(content="original")
        original_hash = doc.content_hash
        original_updated = doc.updated_at

        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)

        doc.update_content("new content")

        assert doc.content == "new content"
        assert doc.content_hash != original_hash
        assert doc.updated_at > original_updated

    def test_add_metadata(self):
        """Adding metadata works."""
        doc = Document(content="test", metadata={"key1": "value1"})
        doc.add_metadata(key2="value2", key3="value3")

        assert doc.metadata["key1"] == "value1"
        assert doc.metadata["key2"] == "value2"
        assert doc.metadata["key3"] == "value3"

    def test_to_from_dict(self):
        """Document serialization/deserialization works."""
        doc = Document(
            content="Test content",
            source_type=SourceType.MARKDOWN,
            source_uri="test.md",
            title="Test",
            metadata={"key": "value"},
            language="en",
        )

        data = doc.to_dict()
        restored = Document.from_dict(data)

        assert restored.doc_id == doc.doc_id
        assert restored.content == doc.content
        assert restored.source_type == doc.source_type
        assert restored.title == doc.title
        assert restored.metadata == doc.metadata

    def test_to_from_json(self):
        """JSON serialization works."""
        doc = Document(
            content="JSON test",
            source_type=SourceType.JSON,
            title="JSON Doc",
        )

        json_str = doc.to_json()
        restored = Document.from_json(json_str)

        assert restored.content == doc.content
        assert restored.source_type == SourceType.JSON

    def test_repr(self):
        """String representation is useful."""
        doc = Document(content="test")
        repr_str = repr(doc)

        assert "Document" in repr_str
        assert "doc_id" in repr_str

    def test_acl_in_document(self):
        """ACL is properly attached to document."""
        acl = AccessControl(owner_id="user1", public=False)
        doc = Document(content="private content", acl=acl)

        assert doc.acl.owner_id == "user1"
        assert doc.acl.can_access(user_id="user1")
        assert not doc.acl.can_access(user_id="user2")


class TestCreateDocument:
    """Tests for create_document factory."""

    def test_basic_creation(self):
        """Factory creates document with defaults."""
        doc = create_document("Hello, world!")

        assert doc.content == "Hello, world!"
        assert doc.source_type == SourceType.UNKNOWN

    def test_with_source_type_string(self):
        """Factory accepts source type as string."""
        doc = create_document("test", source_type="pdf")
        assert doc.source_type == SourceType.PDF

    def test_with_owner(self):
        """Factory sets owner in ACL."""
        doc = create_document("test", owner_id="user123")
        assert doc.acl.owner_id == "user123"

    def test_with_public(self):
        """Factory sets public flag."""
        doc = create_document("test", public=True)
        assert doc.acl.public is True

    def test_with_metadata(self):
        """Factory accepts metadata kwargs."""
        doc = create_document(
            "test",
            title="Test Doc",
            author="John",
            category="test",
        )

        assert doc.title == "Test Doc"
        assert doc.metadata["author"] == "John"
        assert doc.metadata["category"] == "test"
