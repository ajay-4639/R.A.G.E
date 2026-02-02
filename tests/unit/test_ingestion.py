"""Unit tests for ingestion steps."""

import pytest
import json
import csv
from pathlib import Path

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.registry import StepRegistry, get_registry
from rag_os.models.document import Document, SourceType
from rag_os.steps.ingestion import (
    BaseIngestionStep,
    IngestionConfig,
    TextFileIngestionStep,
    MarkdownIngestionStep,
    JSONIngestionStep,
    CSVIngestionStep,
)


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello, world!\nThis is a test file.")
    return file_path


@pytest.fixture
def temp_markdown_file(tmp_path: Path) -> Path:
    """Create a temporary markdown file."""
    file_path = tmp_path / "test.md"
    file_path.write_text("# Test Document\n\nThis is **markdown** content.")
    return file_path


@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    """Create a temporary JSON file."""
    file_path = tmp_path / "test.json"
    data = {"title": "Test", "content": "JSON content here", "tags": ["a", "b"]}
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file."""
    file_path = tmp_path / "test.csv"
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "description", "value"])
        writer.writerow(["Item 1", "First item description", "100"])
        writer.writerow(["Item 2", "Second item description", "200"])
    return file_path


@pytest.fixture
def temp_dir_with_files(tmp_path: Path) -> Path:
    """Create a directory with multiple files."""
    (tmp_path / "file1.txt").write_text("Content 1")
    (tmp_path / "file2.txt").write_text("Content 2")
    (tmp_path / "file3.md").write_text("# Markdown")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file4.txt").write_text("Nested content")
    return tmp_path


class TestIngestionConfig:
    """Tests for IngestionConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = IngestionConfig()
        assert config.encoding == "utf-8"
        assert config.recursive is True
        assert config.skip_errors is False

    def test_custom_config(self):
        """Custom config values work."""
        config = IngestionConfig(
            source_path="/path/to/files",
            encoding="latin-1",
            recursive=False,
            owner_id="user123",
        )
        assert config.source_path == "/path/to/files"
        assert config.encoding == "latin-1"
        assert config.owner_id == "user123"


class TestBaseIngestionStep:
    """Tests for BaseIngestionStep base class."""

    def test_pre_hooks(self):
        """Pre-processing hooks work."""

        class TestIngestor(BaseIngestionStep):
            def ingest(self, source_config=None):
                return []

        step = TestIngestor(step_id="test", config={})

        # Add a hook that uppercases content
        step.add_pre_hook("uppercase", lambda c, m: c.upper())

        result = step.apply_pre_hooks("hello", {})
        assert result == "HELLO"

    def test_remove_hook(self):
        """Removing hooks works."""

        class TestIngestor(BaseIngestionStep):
            def ingest(self, source_config=None):
                return []

        step = TestIngestor(step_id="test", config={})
        step.add_pre_hook("test", lambda c, m: c.upper())

        assert step.remove_pre_hook("test") is True
        assert step.remove_pre_hook("nonexistent") is False

    def test_create_document(self):
        """Document creation helper works."""

        class TestIngestor(BaseIngestionStep):
            def ingest(self, source_config=None):
                return []

        step = TestIngestor(
            step_id="test",
            config={"owner_id": "user123", "public": False},
        )

        doc = step.create_document(
            content="Test content",
            source_uri="/path/to/file",
            source_type=SourceType.TEXT,
            title="Test Doc",
        )

        assert doc.content == "Test content"
        assert doc.source_type == SourceType.TEXT
        assert doc.acl.owner_id == "user123"


class TestTextFileIngestionStep:
    """Tests for TextFileIngestionStep."""

    def test_ingest_single_file(self, temp_text_file: Path):
        """Ingests a single text file."""
        step = TextFileIngestionStep(
            step_id="ingest",
            config={"source_path": str(temp_text_file)},
        )

        docs = step.ingest()

        assert len(docs) == 1
        assert "Hello, world!" in docs[0].content
        assert docs[0].source_type == SourceType.TEXT

    def test_ingest_directory(self, temp_dir_with_files: Path):
        """Ingests all text files in directory."""
        step = TextFileIngestionStep(
            step_id="ingest",
            config={
                "source_path": str(temp_dir_with_files),
                "file_patterns": ["*.txt"],
                "recursive": True,
            },
        )

        docs = step.ingest()

        assert len(docs) == 3  # file1.txt, file2.txt, subdir/file4.txt

    def test_ingest_non_recursive(self, temp_dir_with_files: Path):
        """Non-recursive mode only gets top-level files."""
        step = TextFileIngestionStep(
            step_id="ingest",
            config={
                "source_path": str(temp_dir_with_files),
                "file_patterns": ["*.txt"],
                "recursive": False,
            },
        )

        docs = step.ingest()

        assert len(docs) == 2  # file1.txt, file2.txt only

    def test_execute_with_context(self, temp_text_file: Path):
        """Execute method works with StepContext."""
        step = TextFileIngestionStep(step_id="ingest", config={})

        context = StepContext(data=str(temp_text_file))
        result = step.execute(context)

        assert result.success
        assert len(result.output) == 1
        assert result.metadata["document_count"] == 1

    def test_file_metadata_extracted(self, temp_text_file: Path):
        """File metadata is extracted."""
        step = TextFileIngestionStep(
            step_id="ingest",
            config={"source_path": str(temp_text_file), "extract_metadata": True},
        )

        docs = step.ingest()

        assert docs[0].metadata.get("file_name") == "test.txt"
        assert docs[0].metadata.get("file_extension") == ".txt"
        assert "file_size" in docs[0].metadata

class TestMarkdownIngestionStep:
    """Tests for MarkdownIngestionStep."""

    def test_ingest_markdown(self, temp_markdown_file: Path):
        """Ingests markdown files."""
        step = MarkdownIngestionStep(
            step_id="ingest",
            config={"source_path": str(temp_markdown_file)},
        )

        docs = step.ingest()

        assert len(docs) == 1
        assert docs[0].source_type == SourceType.MARKDOWN
        assert docs[0].title == "Test Document"  # Extracted from heading

class TestJSONIngestionStep:
    """Tests for JSONIngestionStep."""

    def test_ingest_json(self, temp_json_file: Path):
        """Ingests JSON files."""
        step = JSONIngestionStep(
            step_id="ingest",
            config={"source_path": str(temp_json_file)},
        )

        docs = step.ingest()

        assert len(docs) == 1
        assert docs[0].source_type == SourceType.JSON

    def test_extract_text_field(self, tmp_path: Path):
        """Extracts specific text field from JSON."""
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps({"text": "Main content", "other": "ignored"}))

        step = JSONIngestionStep(
            step_id="ingest",
            config={"source_path": str(file_path), "text_field": "text"},
        )

        docs = step.ingest()

        assert docs[0].content == "Main content"

    def test_json_array(self, tmp_path: Path):
        """Handles JSON arrays."""
        file_path = tmp_path / "array.json"
        file_path.write_text(json.dumps([{"text": "First"}, {"text": "Second"}]))

        step = JSONIngestionStep(
            step_id="ingest",
            config={"source_path": str(file_path), "text_field": "text"},
        )

        docs = step.ingest()

        assert "First" in docs[0].content
        assert "Second" in docs[0].content


class TestCSVIngestionStep:
    """Tests for CSVIngestionStep."""

    def test_ingest_csv_one_per_row(self, temp_csv_file: Path):
        """Ingests CSV with one document per row."""
        step = CSVIngestionStep(
            step_id="ingest",
            config={
                "source_path": str(temp_csv_file),
                "one_doc_per_row": True,
            },
        )

        docs = step.ingest()

        assert len(docs) == 2  # 2 data rows
        assert docs[0].source_type == SourceType.CSV

    def test_ingest_csv_single_doc(self, temp_csv_file: Path):
        """Ingests CSV as single document."""
        step = CSVIngestionStep(
            step_id="ingest",
            config={
                "source_path": str(temp_csv_file),
                "one_doc_per_row": False,
            },
        )

        docs = step.ingest()

        assert len(docs) == 1
        assert "Item 1" in docs[0].content
        assert "Item 2" in docs[0].content

    def test_specific_columns(self, temp_csv_file: Path):
        """Extracts only specific columns."""
        step = CSVIngestionStep(
            step_id="ingest",
            config={
                "source_path": str(temp_csv_file),
                "text_columns": ["description"],
                "one_doc_per_row": True,
            },
        )

        docs = step.ingest()

        assert "First item description" in docs[0].content
        assert "Item 1" not in docs[0].content  # name column not included
