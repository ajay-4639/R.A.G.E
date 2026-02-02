"""File-based ingestion steps."""

import os
from pathlib import Path
from typing import Any
import fnmatch

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.document import Document, SourceType
from rag_os.steps.ingestion.base import BaseIngestionStep


@register_step(
    name="TextFileIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests plain text files (.txt)",
    version="1.0.0",
)
class TextFileIngestionStep(BaseIngestionStep):
    """Ingestion step for plain text files."""

    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """Ingest text files from the configured source."""
        config = self.get_config()

        # Merge runtime config
        if source_config:
            if "source_path" in source_config:
                config.source_path = source_config["source_path"]

        if not config.source_path:
            raise ValueError("source_path is required for file ingestion")

        source_path = Path(config.source_path)
        documents: list[Document] = []

        if source_path.is_file():
            # Single file
            doc = self._ingest_file(source_path, config.encoding)
            if doc:
                documents.append(doc)
        elif source_path.is_dir():
            # Directory - find matching files
            files = self._find_files(
                source_path,
                config.file_patterns,
                config.recursive,
            )
            for file_path in files:
                try:
                    doc = self._ingest_file(file_path, config.encoding)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    if not config.skip_errors:
                        raise
                    # Log and skip on error

        return documents

    def _find_files(
        self,
        directory: Path,
        patterns: list[str],
        recursive: bool,
    ) -> list[Path]:
        """Find files matching patterns in directory."""
        files: list[Path] = []

        if recursive:
            for root, _, filenames in os.walk(directory):
                root_path = Path(root)
                for filename in filenames:
                    if self._matches_patterns(filename, patterns):
                        files.append(root_path / filename)
        else:
            for item in directory.iterdir():
                if item.is_file() and self._matches_patterns(item.name, patterns):
                    files.append(item)

        return sorted(files)

    def _matches_patterns(self, filename: str, patterns: list[str]) -> bool:
        """Check if filename matches any of the patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _ingest_file(self, file_path: Path, encoding: str) -> Document | None:
        """Ingest a single text file."""
        config = self.get_config()

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            if config.skip_errors:
                return None
            raise ValueError(f"File {file_path} exceeds max size ({file_size_mb:.1f}MB > {config.max_file_size_mb}MB)")

        # Read content
        content = file_path.read_text(encoding=encoding)

        # Extract metadata
        metadata: dict[str, Any] = {}
        if config.extract_metadata:
            stat = file_path.stat()
            metadata = {
                "file_name": file_path.name,
                "file_size": stat.st_size,
                "file_extension": file_path.suffix,
            }

        return self.create_document(
            content=content,
            source_uri=str(file_path.absolute()),
            source_type=SourceType.TEXT,
            title=file_path.stem,
            metadata=metadata,
        )


@register_step(
    name="MarkdownIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests Markdown files (.md)",
    version="1.0.0",
)
class MarkdownIngestionStep(TextFileIngestionStep):
    """Ingestion step for Markdown files."""

    def _ingest_file(self, file_path: Path, encoding: str) -> Document | None:
        """Ingest a single Markdown file."""
        doc = super()._ingest_file(file_path, encoding)
        if doc:
            doc.source_type = SourceType.MARKDOWN
            # Extract title from first heading if present
            lines = doc.content.split("\n")
            for line in lines:
                if line.startswith("# "):
                    doc.title = line[2:].strip()
                    break
        return doc


@register_step(
    name="JSONIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests JSON files",
    version="1.0.0",
)
class JSONIngestionStep(TextFileIngestionStep):
    """Ingestion step for JSON files."""

    def _ingest_file(self, file_path: Path, encoding: str) -> Document | None:
        """Ingest a single JSON file."""
        import json

        config = self.get_config()

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            if config.skip_errors:
                return None
            raise ValueError(f"File {file_path} exceeds max size")

        # Read and parse JSON
        content = file_path.read_text(encoding=encoding)
        data = json.loads(content)

        # Extract text content from JSON
        # Default: stringify the whole thing
        # Can be customized via config
        text_field = self.config.get("text_field")
        if text_field and isinstance(data, dict):
            text_content = str(data.get(text_field, ""))
        elif isinstance(data, list):
            # List of objects - extract text from each
            text_content = "\n".join(
                str(item.get(text_field, item) if text_field and isinstance(item, dict) else item)
                for item in data
            )
        else:
            text_content = json.dumps(data, indent=2)

        metadata: dict[str, Any] = {"original_structure": type(data).__name__}
        if config.extract_metadata:
            stat = file_path.stat()
            metadata.update({
                "file_name": file_path.name,
                "file_size": stat.st_size,
            })

        return self.create_document(
            content=text_content,
            source_uri=str(file_path.absolute()),
            source_type=SourceType.JSON,
            title=file_path.stem,
            metadata=metadata,
        )


@register_step(
    name="CSVIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests CSV files",
    version="1.0.0",
)
class CSVIngestionStep(BaseIngestionStep):
    """Ingestion step for CSV files."""

    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """Ingest CSV files."""
        import csv

        config = self.get_config()

        if source_config and "source_path" in source_config:
            config.source_path = source_config["source_path"]

        if not config.source_path:
            raise ValueError("source_path is required")

        source_path = Path(config.source_path)
        documents: list[Document] = []

        # Get CSV-specific config
        text_columns = self.config.get("text_columns", [])
        delimiter = self.config.get("delimiter", ",")
        one_doc_per_row = self.config.get("one_doc_per_row", True)

        with open(source_path, encoding=config.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            if one_doc_per_row:
                # Create one document per row
                for i, row in enumerate(reader):
                    if text_columns:
                        content = " ".join(str(row.get(col, "")) for col in text_columns if col in row)
                    else:
                        content = " ".join(str(v) for v in row.values())

                    doc = self.create_document(
                        content=content,
                        source_uri=f"{source_path}#row{i}",
                        source_type=SourceType.CSV,
                        title=f"{source_path.stem}_row{i}",
                        metadata={"row_index": i, "columns": list(row.keys())},
                    )
                    documents.append(doc)
            else:
                # Create one document for entire file
                rows = list(reader)
                if text_columns:
                    content = "\n".join(
                        " ".join(str(row.get(col, "")) for col in text_columns if col in row)
                        for row in rows
                    )
                else:
                    content = "\n".join(
                        " ".join(str(v) for v in row.values())
                        for row in rows
                    )

                doc = self.create_document(
                    content=content,
                    source_uri=str(source_path.absolute()),
                    source_type=SourceType.CSV,
                    title=source_path.stem,
                    metadata={"row_count": len(rows)},
                )
                documents.append(doc)

        return documents


# Update the ingestion __init__.py to include these
