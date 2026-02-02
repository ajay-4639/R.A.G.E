"""RAG OS Client - High-level interface for RAG pipelines."""

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from rag_os.core.spec import PipelineSpec
from rag_os.core.executor import PipelineExecutor, PipelineResult
from rag_os.storage import MemoryStorage, StorageConfig
from rag_os.tracing import Tracer, get_tracer


@dataclass
class RAGClientConfig:
    """Configuration for RAG client.

    Attributes:
        storage_config: Storage backend configuration
        enable_tracing: Whether to enable tracing
        enable_metrics: Whether to enable metrics
        default_timeout_ms: Default timeout for operations
        cache_enabled: Whether to enable response caching
        cache_ttl_seconds: Cache TTL
    """
    storage_config: StorageConfig | None = None
    enable_tracing: bool = True
    enable_metrics: bool = True
    default_timeout_ms: int = 30000
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600


class RAGClient:
    """High-level client for RAG OS.

    Provides a simple interface for:
    - Loading and running pipelines
    - Managing documents
    - Query execution
    - Caching and storage

    Usage:
        client = RAGClient()
        result = client.query("What is the capital of France?", pipeline="qa")
    """

    def __init__(self, config: RAGClientConfig | None = None):
        """Initialize RAG client.

        Args:
            config: Client configuration
        """
        self.config = config or RAGClientConfig()
        self._pipelines: dict[str, PipelineSpec] = {}
        self._engines: dict[str, PipelineExecutor] = {}
        self._storage = MemoryStorage(self.config.storage_config)
        self._tracer = get_tracer() if self.config.enable_tracing else None
        self._hooks: dict[str, list[Callable]] = {
            "pre_query": [],
            "post_query": [],
            "pre_step": [],
            "post_step": [],
        }

    def load_pipeline(
        self,
        source: str | Path | dict | PipelineSpec,
        name: str | None = None,
    ) -> str:
        """Load a pipeline from various sources.

        Args:
            source: Pipeline source (file path, dict, or PipelineSpec)
            name: Optional name override

        Returns:
            Pipeline name
        """
        if isinstance(source, PipelineSpec):
            spec = source
        elif isinstance(source, dict):
            spec = PipelineSpec.from_dict(source)
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix in ('.yaml', '.yml'):
                spec = PipelineSpec.from_yaml(path)
            elif path.suffix == '.json':
                spec = PipelineSpec.from_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        pipeline_name = name or spec.name
        self._pipelines[pipeline_name] = spec
        self._engines[pipeline_name] = PipelineExecutor()

        return pipeline_name

    def query(
        self,
        query: str,
        pipeline: str = "default",
        context: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Execute a query against a pipeline.

        Args:
            query: The query string
            pipeline: Pipeline name to use
            context: Additional context data
            timeout_ms: Operation timeout

        Returns:
            Query result dictionary
        """
        if pipeline not in self._pipelines:
            raise ValueError(f"Pipeline '{pipeline}' not loaded. Use load_pipeline() first.")

        # Check cache
        cache_key = f"query:{pipeline}:{hash(query)}"
        if self.config.cache_enabled:
            cached = self._storage.get(cache_key)
            if cached is not None:
                return cached

        # Run pre-query hooks
        for hook in self._hooks["pre_query"]:
            hook(query, pipeline, context)

        # Build context
        ctx_data = {"query": query}
        if context:
            ctx_data.update(context)

        # Execute pipeline
        spec = self._pipelines[pipeline]
        executor = self._engines[pipeline]

        if self._tracer:
            with self._tracer.trace(f"query:{pipeline}") as span:
                span.set_attribute("query", query[:100])
                result = executor.execute(spec, initial_data=ctx_data)
        else:
            result = executor.execute(spec, initial_data=ctx_data)

        # Build response
        response = {
            "query": query,
            "answer": self._extract_answer(result),
            "pipeline": pipeline,
            "success": result.success,
            "latency_ms": result.total_latency_ms,
            "metadata": result.metadata,
        }

        # Run post-query hooks
        for hook in self._hooks["post_query"]:
            hook(query, response)

        # Cache result
        if self.config.cache_enabled and result.success:
            self._storage.set(cache_key, response, self.config.cache_ttl_seconds)

        return response

    def _extract_answer(self, result: PipelineResult) -> str:
        """Extract answer from pipeline result."""
        if not result.success:
            return f"Error: {result.error}"

        # Try to find the answer in the result
        output = result.output

        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            # Look for common answer keys
            for key in ["answer", "content", "response", "text", "output"]:
                if key in output:
                    val = output[key]
                    if isinstance(val, str):
                        return val
            return str(output)
        if hasattr(output, "content"):
            return str(output.content)

        return str(output)

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        pipeline: str = "default",
    ) -> int:
        """Add documents to the pipeline's index.

        Args:
            documents: List of documents to add
            pipeline: Pipeline to add documents to

        Returns:
            Number of documents added
        """
        # Store documents
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{i}")
            self._storage.set(f"doc:{pipeline}:{doc_id}", doc)

        return len(documents)

    def get_pipelines(self) -> list[str]:
        """Get list of loaded pipeline names."""
        return list(self._pipelines.keys())

    def list_pipelines(self) -> list[str]:
        """Alias for get_pipelines() for API compatibility."""
        return self.get_pipelines()

    def unload_pipeline(self, name: str) -> None:
        """Unload a pipeline.

        Args:
            name: Pipeline name to unload
        """
        if name not in self._pipelines:
            raise ValueError(f"Pipeline '{name}' not found")
        del self._pipelines[name]
        if name in self._engines:
            del self._engines[name]

    def get_pipeline_info(self, name: str) -> dict[str, Any] | None:
        """Get information about a pipeline.

        Args:
            name: Pipeline name

        Returns:
            Pipeline information dictionary or None if not found
        """
        if name not in self._pipelines:
            return None

        spec = self._pipelines[name]
        return {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "steps": [
                {
                    "id": s.step_id,
                    "step_class": s.step_class,
                    "enabled": s.enabled,
                }
                for s in spec.steps
            ],
        }

    def register_hook(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register a hook callback.

        Args:
            event: Event name (pre_query, post_query, pre_step, post_step)
            callback: Callback function
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}")
        self._hooks[event].append(callback)

    def clear_cache(self) -> int:
        """Clear the response cache.

        Returns:
            Number of items cleared
        """
        return self._storage.clear()

    def close(self) -> None:
        """Close the client and release resources."""
        self._storage.clear()
        self._pipelines.clear()
        self._engines.clear()


def create_client(config: RAGClientConfig | None = None) -> RAGClient:
    """Create a new RAG client.

    Args:
        config: Client configuration

    Returns:
        RAGClient instance
    """
    return RAGClient(config)
