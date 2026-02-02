"""Embedding provider implementations for RAG OS."""

import math
import hashlib
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.steps.embedding.base import BaseEmbeddingStep


@register_step(
    name="MockEmbeddingStep",
    step_type=StepType.EMBEDDING,
    description="Mock embedding step for testing",
    version="1.0.0",
)
class MockEmbeddingStep(BaseEmbeddingStep):
    """Mock embedding step for testing purposes.

    Generates deterministic pseudo-embeddings based on text content.
    Useful for testing without API calls.
    """

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.get("model_name", "mock-embedding-model")

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.config.get("dimensions", 384)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic mock embeddings.

        Creates embeddings based on text hash for reproducibility.
        """
        dims = self.dimensions
        embeddings: list[list[float]] = []

        for text in texts:
            # Create deterministic embedding from text hash
            embedding = self._text_to_embedding(text, dims)
            if self.get_config().normalize:
                embedding = self._normalize(embedding)
            embeddings.append(embedding)

        return embeddings

    def _text_to_embedding(self, text: str, dims: int) -> list[float]:
        """Convert text to a deterministic embedding vector."""
        # Use hash for determinism
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Generate embedding from hash
        embedding: list[float] = []
        for i in range(dims):
            # Use different parts of hash for each dimension
            idx = (i * 2) % len(text_hash)
            val = int(text_hash[idx : idx + 2], 16) / 255.0 - 0.5
            # Add some variation based on position and text length
            val += math.sin(i * 0.1 + len(text) * 0.01) * 0.1
            embedding.append(val)

        return embedding

    def _normalize(self, embedding: list[float]) -> list[float]:
        """Normalize embedding to unit length."""
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding


@register_step(
    name="OpenAIEmbeddingStep",
    step_type=StepType.EMBEDDING,
    description="OpenAI embedding API integration",
    version="1.0.0",
)
class OpenAIEmbeddingStep(BaseEmbeddingStep):
    """Embedding step using OpenAI's embedding API.

    Supports text-embedding-3-small, text-embedding-3-large,
    and text-embedding-ada-002 models.
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._client = None
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.get("model_name", "text-embedding-3-small")

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Check config first
        config_dims = self.config.get("dimensions", 0)
        if config_dims > 0:
            return config_dims

        # Use default for model
        return self.MODEL_DIMENSIONS.get(self.model_name, 1536)

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )

            api_key = self.config.get("api_key")
            api_base = self.config.get("api_base")

            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                kwargs["base_url"] = api_base

            self._client = OpenAI(**kwargs)

        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        if not texts:
            return []

        client = self._get_client()
        config = self.get_config()

        # Build request params
        params: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        # text-embedding-3 models support custom dimensions
        if "text-embedding-3" in self.model_name and config.dimensions > 0:
            params["dimensions"] = config.dimensions

        # Make API call with retry
        for attempt in range(config.max_retries):
            try:
                response = client.embeddings.create(**params)

                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]

                # Store actual dimensions
                if embeddings:
                    self._dimensions = len(embeddings[0])

                return embeddings

            except Exception as e:
                if attempt == config.max_retries - 1:
                    raise
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)

        return []

    def embed_query(self, query: str) -> list[float]:
        """Embed a query (same as document for OpenAI)."""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []


@register_step(
    name="LocalEmbeddingStep",
    step_type=StepType.EMBEDDING,
    description="Local embedding using sentence-transformers",
    version="1.0.0",
)
class LocalEmbeddingStep(BaseEmbeddingStep):
    """Embedding step using local sentence-transformers models.

    Runs entirely locally without API calls.
    """

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._model = None
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.get("model_name", "all-MiniLM-L6-v2")

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Check config
        config_dims = self.config.get("dimensions", 0)
        if config_dims > 0:
            return config_dims

        # Default for common models
        model_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
        }
        return model_dims.get(self.model_name, 384)

    def _get_model(self):
        """Load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )

            self._model = SentenceTransformer(self.model_name)
            # Get actual dimensions
            self._dimensions = self._model.get_sentence_embedding_dimension()

        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local model."""
        if not texts:
            return []

        model = self._get_model()
        config = self.get_config()

        # Encode with sentence-transformers
        embeddings = model.encode(
            texts,
            normalize_embeddings=config.normalize,
            show_progress_bar=config.show_progress,
            batch_size=config.batch_size,
        )

        # Convert numpy array to list
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query."""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []


@register_step(
    name="CohereEmbeddingStep",
    step_type=StepType.EMBEDDING,
    description="Cohere embedding API integration",
    version="1.0.0",
)
class CohereEmbeddingStep(BaseEmbeddingStep):
    """Embedding step using Cohere's embedding API.

    Supports embed-english-v3.0, embed-multilingual-v3.0, etc.
    """

    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    def __init__(self, step_id: str, config: dict[str, Any] | None = None):
        super().__init__(step_id, config=config)
        self._client = None
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.get("model_name", "embed-english-v3.0")

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        config_dims = self.config.get("dimensions", 0)
        if config_dims > 0:
            return config_dims

        return self.MODEL_DIMENSIONS.get(self.model_name, 1024)

    def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "cohere package required. Install with: pip install cohere"
                )

            api_key = self.config.get("api_key")
            self._client = cohere.Client(api_key) if api_key else cohere.Client()

        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Cohere API."""
        if not texts:
            return []

        client = self._get_client()
        config = self.get_config()

        for attempt in range(config.max_retries):
            try:
                response = client.embed(
                    texts=texts,
                    model=self.model_name,
                    input_type="search_document",
                )

                embeddings = response.embeddings
                if embeddings:
                    self._dimensions = len(embeddings[0])

                return embeddings

            except Exception as e:
                if attempt == config.max_retries - 1:
                    raise
                import time
                time.sleep(2 ** attempt)

        return []

    def embed_query(self, query: str) -> list[float]:
        """Embed a query (uses search_query input type)."""
        client = self._get_client()
        config = self.get_config()

        for attempt in range(config.max_retries):
            try:
                response = client.embed(
                    texts=[query],
                    model=self.model_name,
                    input_type="search_query",
                )

                return response.embeddings[0] if response.embeddings else []

            except Exception as e:
                if attempt == config.max_retries - 1:
                    raise
                import time
                time.sleep(2 ** attempt)

        return []
