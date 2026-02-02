"""PDF ingestion step for RAG OS."""

from pathlib import Path
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.document import Document, SourceType
from rag_os.steps.ingestion.base import BaseIngestionStep


@register_step(
    name="PDFIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests PDF files",
    version="1.0.0",
)
class PDFIngestionStep(BaseIngestionStep):
    """Ingestion step for PDF files.

    Uses PyMuPDF (fitz) for text extraction. Supports both
    file path input and raw bytes input for upload scenarios.

    Config options:
        source_path: Path to a PDF file or directory containing PDFs
        file_bytes: Raw PDF bytes (for upload scenarios)
        file_name: Original filename when using file_bytes
    """

    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """Ingest PDF files from the configured source."""
        config = self.get_config()

        if source_config:
            if "source_path" in source_config:
                config.source_path = source_config["source_path"]

        # Check for raw bytes input (upload scenario)
        file_bytes = None
        file_name = "uploaded.pdf"
        if source_config:
            file_bytes = source_config.get("file_bytes")
            file_name = source_config.get("file_name", file_name)

        if file_bytes:
            return [self._ingest_bytes(file_bytes, file_name)]

        if not config.source_path:
            raise ValueError("source_path or file_bytes is required for PDF ingestion")

        source_path = Path(config.source_path)
        documents: list[Document] = []

        if source_path.is_file():
            doc = self._ingest_file(source_path)
            if doc:
                documents.append(doc)
        elif source_path.is_dir():
            for pdf_file in sorted(source_path.rglob("*.pdf") if config.recursive else source_path.glob("*.pdf")):
                try:
                    doc = self._ingest_file(pdf_file)
                    if doc:
                        documents.append(doc)
                except Exception:
                    if not config.skip_errors:
                        raise

        return documents

    def _ingest_file(self, file_path: Path) -> Document | None:
        """Ingest a single PDF file from disk."""
        config = self.get_config()

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            if config.skip_errors:
                return None
            raise ValueError(
                f"File {file_path} exceeds max size "
                f"({file_size_mb:.1f}MB > {config.max_file_size_mb}MB)"
            )

        content, page_count = self._extract_text_from_path(file_path)

        metadata: dict[str, Any] = {}
        if config.extract_metadata:
            stat = file_path.stat()
            metadata = {
                "file_name": file_path.name,
                "file_size": stat.st_size,
                "file_extension": ".pdf",
                "page_count": page_count,
            }

        return self.create_document(
            content=content,
            source_uri=str(file_path.absolute()),
            source_type=SourceType.PDF,
            title=file_path.stem,
            metadata=metadata,
        )

    def _ingest_bytes(self, file_bytes: bytes, file_name: str) -> Document:
        """Ingest a PDF from raw bytes (upload scenario)."""
        config = self.get_config()

        # Check size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            raise ValueError(
                f"File exceeds max size "
                f"({file_size_mb:.1f}MB > {config.max_file_size_mb}MB)"
            )

        content, page_count = self._extract_text_from_bytes(file_bytes)

        metadata: dict[str, Any] = {
            "file_name": file_name,
            "file_size": len(file_bytes),
            "file_extension": ".pdf",
            "page_count": page_count,
        }

        title = Path(file_name).stem if file_name else "uploaded_pdf"

        return self.create_document(
            content=content,
            source_uri=f"upload://{file_name}",
            source_type=SourceType.PDF,
            title=title,
            metadata=metadata,
        )

    def _extract_text_from_path(self, file_path: Path) -> tuple[str, int]:
        """Extract text from a PDF file path.

        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF ingestion. "
                "Install with: pip install pymupdf"
            )

        doc = fitz.open(str(file_path))
        try:
            return self._extract_from_fitz_doc(doc)
        finally:
            doc.close()

    def _extract_text_from_bytes(self, file_bytes: bytes) -> tuple[str, int]:
        """Extract text from raw PDF bytes.

        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF ingestion. "
                "Install with: pip install pymupdf"
            )

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return self._extract_from_fitz_doc(doc)
        finally:
            doc.close()

    def _extract_from_fitz_doc(self, doc: Any) -> tuple[str, int]:
        """Extract text from a fitz Document object.

        Returns:
            Tuple of (extracted_text, page_count)
        """
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text.strip())

        content = "\n\n".join(pages)
        return content, len(doc)
