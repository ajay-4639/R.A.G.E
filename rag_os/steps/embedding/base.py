"""Base embedding step for RAG OS."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.step import Step
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.models.chunk import Chunk
from rag_os.models.embedding import EmbeddedChunk, EmbeddingConfig


@dataclass
class EmbeddingStepConfig:
    """Configuration for embedding steps.

    Attributes:
        model_name: Name of the embedding model
        dimensions: Expected embedding dimensions (0 = auto-detect)
        batch_size: Batch size for embedding operations
        normalize: Whether to normalize embeddings to unit length
        max_retries: Maximum retries for failed operations
        timeout: Timeout in seconds for API calls
        show_progress: Whether to show progress during embedding
    """
    model_name: str = "text-embedding-3-small"
    dimensions: int = 0
    batch_size: int = 100
    normalize: bool = True
    max_retries: int = 3
    timeout: float = 30.0
    show_progress: bool = False


class BaseEmbeddingStep(Step):
    """Abstract base class for embedding steps.

    Embedding steps take chunks and produce embedded chunks with
    vector representations.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.EMBEDDING,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)

    @property
    def input_schema(self) -> dict[str, Any]:
        """Embedding steps accept List[Chunk] or List[str]."""
        return {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "object", "description": "Chunk object"},
                    {"type": "string", "description": "Text string"},
                ]
            },
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        """Embedding steps produce List[EmbeddedChunk]."""
        return {
            "type": "array",
            "items": {
                "type": "object",
                "description": "EmbeddedChunk with chunk and embedding vector",
            },
        }

    def get_config(self) -> EmbeddingStepConfig:
        """Get typed configuration."""
        return EmbeddingStepConfig(
            model_name=self.config.get("model_name", "text-embedding-3-small"),
            dimensions=self.config.get("dimensions", 0),
            batch_size=self.config.get("batch_size", 100),
            normalize=self.config.get("normalize", True),
            max_retries=self.config.get("max_retries", 3),
            timeout=self.config.get("timeout", 30.0),
            show_progress=self.config.get("show_progress", False),
        )

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        pass

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query.

        Some models use different embeddings for queries vs documents.
        Default implementation just calls embed_texts.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        pass

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks.

        Args:
            chunks: Chunks to embed

        Returns:
            List of EmbeddedChunk with embeddings
        """
        if not chunks:
            return []

        config = self.get_config()

        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks]

        # Batch the embedding
        all_embeddings: list[list[float]] = []
        batch_size = config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed_texts(batch)
            all_embeddings.extend(embeddings)

        # Create EmbeddedChunk objects
        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, embedding in zip(chunks, all_embeddings):
            embedded = EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                model_name=self.model_name,
                dimensions=self.dimensions,
            )
            embedded_chunks.append(embedded)

        return embedded_chunks

    def execute(self, context: StepContext) -> StepResult:
        """Execute the embedding step.

        Accepts either List[Chunk] or List[str] as input.
        """
        import time
        start_time = time.time()

        data = context.data
        if not data:
            return StepResult.ok(output=[], embedded_count=0)

        # Handle different input types
        if isinstance(data, list):
            if all(isinstance(item, Chunk) for item in data):
                # List of Chunks
                chunks = data
            elif all(isinstance(item, str) for item in data):
                # List of strings - create temporary chunks
                chunks = [
                    Chunk(content=text, doc_id="temp", index=i)
                    for i, text in enumerate(data)
                ]
            else:
                return StepResult.fail(
                    error="Input must be List[Chunk] or List[str]",
                    input_type=str(type(data)),
                )
        else:
            return StepResult.fail(
                error="Input must be a list",
                input_type=str(type(data)),
            )

        try:
            embedded_chunks = self.embed_chunks(chunks)
            latency = (time.time() - start_time) * 1000

            return StepResult.ok(
                output=embedded_chunks,
                embedded_count=len(embedded_chunks),
                model_name=self.model_name,
                dimensions=self.dimensions,
            ).with_latency(latency)

        except Exception as e:
            return StepResult.fail(
                error=str(e),
                step_id=self.step_id,
            )
