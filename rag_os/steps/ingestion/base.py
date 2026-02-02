"""Base ingestion step interface."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ConfigDict

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.models.document import Document, SourceType, AccessControl


class IngestionConfig(BaseModel):
    """Configuration for ingestion steps."""

    # Source configuration
    source_path: str | None = Field(default=None, description="Path to source file or directory")
    source_uri: str | None = Field(default=None, description="URI of source")
    file_patterns: list[str] = Field(default_factory=lambda: ["*"], description="File patterns to match")
    recursive: bool = Field(default=True, description="Recursively process directories")

    # Processing options
    encoding: str = Field(default="utf-8", description="Text encoding")
    max_file_size_mb: float = Field(default=100.0, description="Maximum file size in MB")
    skip_errors: bool = Field(default=False, description="Skip files that cause errors")

    # Metadata options
    extract_metadata: bool = Field(default=True, description="Extract metadata from files")
    default_language: str = Field(default="", description="Default language code")

    # Access control
    owner_id: str | None = Field(default=None, description="Owner user ID")
    public: bool = Field(default=False, description="Make documents public")
    team_ids: list[str] = Field(default_factory=list, description="Team IDs with access")

    model_config = ConfigDict(extra="allow")


@dataclass
class PreProcessingHook:
    """Hook for pre-processing content before document creation."""

    name: str
    func: Callable[[str, dict[str, Any]], str]
    enabled: bool = True

    def __call__(self, content: str, metadata: dict[str, Any]) -> str:
        if self.enabled:
            return self.func(content, metadata)
        return content


class BaseIngestionStep(Step):
    """
    Base class for all ingestion steps.

    Ingestion steps are responsible for loading documents from various
    sources (files, URLs, APIs, etc.) and converting them to Document
    objects that can be processed by subsequent pipeline steps.

    Subclasses must implement the `ingest()` method.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.INGESTION,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)
        self._pre_hooks: list[PreProcessingHook] = []
        self._parsed_config: IngestionConfig | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        Ingestion steps accept optional source configuration.

        The input can be:
        - None (use config for source)
        - A string (source path or URI)
        - A dict with source configuration
        """
        return {
            "type": ["object", "string", "null"],
            "properties": {
                "source_path": {"type": "string"},
                "source_uri": {"type": "string"},
            },
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Ingestion steps output a list of Documents."""
        return {
            "type": "array",
            "items": {"type": "object"},
            "description": "List of ingested documents",
        }

    @property
    def config_schema(self) -> dict[str, Any]:
        """Return the config schema based on IngestionConfig."""
        return {
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "source_uri": {"type": "string"},
                "file_patterns": {"type": "array", "items": {"type": "string"}},
                "recursive": {"type": "boolean"},
                "encoding": {"type": "string"},
                "max_file_size_mb": {"type": "number"},
                "skip_errors": {"type": "boolean"},
            },
        }

    def get_config(self) -> IngestionConfig:
        """Get parsed configuration."""
        if self._parsed_config is None:
            self._parsed_config = IngestionConfig(**self.config)
        return self._parsed_config

    def add_pre_hook(
        self,
        name: str,
        func: Callable[[str, dict[str, Any]], str],
    ) -> None:
        """
        Add a pre-processing hook.

        Hooks are applied to content before Document creation.

        Args:
            name: Hook identifier
            func: Function that takes (content, metadata) and returns modified content
        """
        self._pre_hooks.append(PreProcessingHook(name=name, func=func))

    def remove_pre_hook(self, name: str) -> bool:
        """Remove a pre-processing hook by name."""
        for i, hook in enumerate(self._pre_hooks):
            if hook.name == name:
                del self._pre_hooks[i]
                return True
        return False

    def apply_pre_hooks(self, content: str, metadata: dict[str, Any]) -> str:
        """Apply all pre-processing hooks to content."""
        for hook in self._pre_hooks:
            content = hook(content, metadata)
        return content

    def create_document(
        self,
        content: str,
        source_uri: str,
        source_type: SourceType,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Create a Document from ingested content.

        This helper method applies pre-processing hooks and sets up
        proper access control based on configuration.

        Args:
            content: The document content
            source_uri: URI of the source
            source_type: Type of the source
            title: Optional document title
            metadata: Additional metadata

        Returns:
            A new Document instance
        """
        config = self.get_config()
        metadata = metadata or {}

        # Apply pre-processing hooks
        content = self.apply_pre_hooks(content, metadata)

        # Create access control
        acl = AccessControl(
            owner_id=config.owner_id,
            team_ids=config.team_ids,
            public=config.public,
        )

        # Create document
        doc = Document(
            content=content,
            source_type=source_type,
            source_uri=source_uri,
            title=title,
            metadata=metadata,
            acl=acl,
            language=config.default_language,
        )

        return doc

    @abstractmethod
    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """
        Ingest documents from the configured source.

        This is the main method that subclasses must implement.

        Args:
            source_config: Optional runtime source configuration
                          (overrides step config)

        Returns:
            List of ingested Document objects
        """
        ...

    def execute(self, context: StepContext) -> StepResult:
        """
        Execute the ingestion step.

        Extracts source configuration from context and calls ingest().

        Args:
            context: Step context with optional source configuration

        Returns:
            StepResult containing list of Documents
        """
        # Extract source configuration from context
        source_config: dict[str, Any] | None = None

        if context.data is None:
            source_config = None
        elif isinstance(context.data, str):
            # Treat string as source path
            source_config = {"source_path": context.data}
        elif isinstance(context.data, dict):
            source_config = context.data

        try:
            documents = self.ingest(source_config)
            return StepResult.ok(
                documents,
                document_count=len(documents),
                total_characters=sum(doc.content_length for doc in documents),
            )
        except Exception as e:
            return StepResult.fail(
                f"Ingestion failed: {e}",
                error_type=type(e).__name__,
            )

    def validate_source(self, source_path: str | None = None) -> list[str]:
        """
        Validate that the source exists and is accessible.

        Override in subclasses for specific validation logic.

        Args:
            source_path: Path to validate

        Returns:
            List of validation error messages
        """
        return []
