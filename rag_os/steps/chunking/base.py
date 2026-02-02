"""Base chunking step interface."""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.models.document import Document
from rag_os.models.chunk import Chunk


class ChunkingConfig(BaseModel):
    """Configuration for chunking steps."""

    # Size configuration
    chunk_size: int = Field(default=1000, description="Target chunk size (chars or tokens)")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    max_chunk_size: int | None = Field(default=None, description="Maximum chunk size")

    # Behavior options
    preserve_sentences: bool = Field(default=True, description="Try to preserve sentence boundaries")
    preserve_paragraphs: bool = Field(default=False, description="Try to preserve paragraph boundaries")
    strip_whitespace: bool = Field(default=True, description="Strip leading/trailing whitespace")

    # Metadata options
    include_metadata: bool = Field(default=True, description="Include source metadata in chunks")
    add_chunk_headers: bool = Field(default=False, description="Add headers with chunk info")

    model_config = ConfigDict(extra="allow")


class BaseChunkingStep(Step):
    """
    Base class for all chunking steps.

    Chunking steps take documents and split them into smaller chunks
    suitable for embedding and retrieval.

    Subclasses must implement the `chunk_document()` method.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.CHUNKING,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)
        self._parsed_config: ChunkingConfig | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        """Chunking steps accept a list of Documents."""
        return {
            "type": "array",
            "items": {"type": "object"},
            "description": "List of documents to chunk",
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Chunking steps output a list of Chunks."""
        return {
            "type": "array",
            "items": {"type": "object"},
            "description": "List of chunks",
        }

    @property
    def config_schema(self) -> dict[str, Any]:
        """Return the config schema."""
        return {
            "type": "object",
            "properties": {
                "chunk_size": {"type": "integer", "minimum": 1},
                "chunk_overlap": {"type": "integer", "minimum": 0},
                "min_chunk_size": {"type": "integer", "minimum": 1},
            },
        }

    def get_config(self) -> ChunkingConfig:
        """Get parsed configuration."""
        if self._parsed_config is None:
            self._parsed_config = ChunkingConfig(**self.config)
        return self._parsed_config

    @abstractmethod
    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a single document into chunks.

        This is the main method that subclasses must implement.

        Args:
            document: The document to chunk

        Returns:
            List of chunks from this document
        """
        ...

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all chunks from all documents
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def execute(self, context: StepContext) -> StepResult:
        """
        Execute the chunking step.

        Args:
            context: Step context containing documents

        Returns:
            StepResult containing list of Chunks
        """
        # Get documents from context
        documents = context.data

        if documents is None:
            return StepResult.fail("No documents provided")

        if not isinstance(documents, list):
            documents = [documents]

        # Convert dicts to Documents if needed
        processed_docs: list[Document] = []
        for doc in documents:
            if isinstance(doc, Document):
                processed_docs.append(doc)
            elif isinstance(doc, dict):
                processed_docs.append(Document.from_dict(doc))
            else:
                return StepResult.fail(f"Invalid document type: {type(doc)}")

        try:
            chunks = self.chunk_documents(processed_docs)
            return StepResult.ok(
                chunks,
                document_count=len(processed_docs),
                chunk_count=len(chunks),
                avg_chunk_size=sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
            )
        except Exception as e:
            return StepResult.fail(f"Chunking failed: {e}")

    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        index: int,
        start_char: int,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """
        Helper to create a chunk with proper configuration.

        Args:
            content: Chunk content
            doc_id: Parent document ID
            index: Chunk index
            start_char: Starting character position
            metadata: Additional metadata

        Returns:
            New Chunk instance
        """
        config = self.get_config()

        # Strip whitespace if configured
        if config.strip_whitespace:
            content = content.strip()

        # Merge metadata
        chunk_metadata = metadata.copy() if metadata else {}
        if config.include_metadata:
            chunk_metadata["chunking_method"] = self.__class__.__name__
            chunk_metadata["chunk_size_config"] = config.chunk_size

        return Chunk(
            content=content,
            doc_id=doc_id,
            index=index,
            start_char=start_char,
            end_char=start_char + len(content),
            metadata=chunk_metadata,
        )
