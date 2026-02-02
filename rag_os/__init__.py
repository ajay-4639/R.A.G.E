"""RAG OS - A fully customizable RAG Operating System.

RAG OS provides a complete framework for building Retrieval-Augmented Generation
pipelines with:

- Modular pipeline steps (parsing, chunking, embedding, retrieval, generation)
- Flexible storage backends (memory, file-based)
- Built-in tracing and metrics
- Evaluation framework with multiple metrics
- SDK with fluent builders
- Plugin system with hooks
- Security features (validation, rate limiting, secret management)

Example (local mode):
    from rag_os import RAGClient, PipelineBuilder

    client = RAGClient()
    client.load_pipeline(pipeline)
    result = client.query("What is RAG?", pipeline="my-pipeline")

Example (remote mode - Docker hosted):
    from rag_os import RemoteClient

    client = RemoteClient("http://localhost:8000")
    client.create_pipeline({"name": "my-rag", "version": "1.0.0", "steps": [...]})
    result = client.query("What is RAG?", pipeline="my-rag")
    print(result.answer)
"""

__version__ = "0.1.0"

# Core pipeline components
from rag_os.core.spec import PipelineSpec, StepSpec
from rag_os.core.executor import PipelineExecutor, PipelineResult
from rag_os.core.context import StepContext
from rag_os.core.registry import StepRegistry

# SDK - High-level API
from rag_os.sdk.client import RAGClient, RAGClientConfig, create_client
from rag_os.sdk.builder import PipelineBuilder, StepBuilder
from rag_os.sdk.session import RAGSession, SessionConfig
from rag_os.sdk.remote import RemoteClient, QueryResult as RemoteQueryResult, RemoteClientError

# Storage backends
from rag_os.storage.base import BaseStorage, BaseDocumentStorage
from rag_os.storage.memory import MemoryStorage, MemoryDocumentStorage
from rag_os.storage.file import FileStorage, FileDocumentStorage

# Tracing and metrics
from rag_os.tracing.tracer import Tracer, Span, get_tracer
from rag_os.tracing.metrics import MetricsCollector, Counter, Gauge, Histogram

# Evaluation
from rag_os.eval.evaluator import Evaluator, EvalResult, EvalConfig
from rag_os.eval.metrics import RetrievalMetrics, AnswerMetrics

# Plugins
from rag_os.plugins.base import Plugin, PluginInfo, PluginManager
from rag_os.plugins.hooks import HookType, HookManager

# Security
from rag_os.security.validation import InputValidator, ValidationRule, sanitize_input
from rag_os.security.rate_limit import RateLimiter, RateLimitConfig
from rag_os.security.secrets import SecretManager, mask_secrets

__all__ = [
    # Version
    "__version__",
    # Core
    "PipelineSpec",
    "StepSpec",
    "PipelineExecutor",
    "PipelineResult",
    "StepContext",
    "StepRegistry",
    # SDK
    "RAGClient",
    "RAGClientConfig",
    "create_client",
    "PipelineBuilder",
    "StepBuilder",
    "RAGSession",
    "SessionConfig",
    # Remote
    "RemoteClient",
    "RemoteQueryResult",
    "RemoteClientError",
    # Storage
    "BaseStorage",
    "BaseDocumentStorage",
    "MemoryStorage",
    "MemoryDocumentStorage",
    "FileStorage",
    "FileDocumentStorage",
    # Tracing
    "Tracer",
    "Span",
    "get_tracer",
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    # Evaluation
    "Evaluator",
    "EvalResult",
    "EvalConfig",
    "RetrievalMetrics",
    "AnswerMetrics",
    # Plugins
    "Plugin",
    "PluginInfo",
    "PluginManager",
    "HookType",
    "HookManager",
    # Security
    "InputValidator",
    "ValidationRule",
    "sanitize_input",
    "RateLimiter",
    "RateLimitConfig",
    "SecretManager",
    "mask_secrets",
]
