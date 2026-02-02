"""Core components of RAG OS."""

from rag_os.core.types import StepType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult
from rag_os.core.step import Step
from rag_os.core.spec import StepSpec, PipelineSpec
from rag_os.core.registry import StepRegistry, StepMetadata, register_step, get_registry
from rag_os.core.validator import (
    PipelineValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_pipeline,
)
from rag_os.core.runtime import (
    RuntimeContext,
    ExecutionMode,
    FailureMode,
    TokenBudget,
    CostBudget,
    RBACScope,
)
from rag_os.core.executor import (
    PipelineExecutor,
    PipelineResult,
    StepExecutionResult,
    execute_pipeline,
)
from rag_os.core.errors import (
    RAGOSError,
    StepError,
    PipelineError,
    ProviderError,
    BudgetExceededError,
    ConfigurationError,
    ValidationError,
    ErrorCode,
    RetryPolicy,
)
from rag_os.core.async_executor import (
    AsyncPipelineExecutor,
    DryRunExecutor,
    BatchResult,
    execute_pipeline_async,
    execute_batch,
)

__all__ = [
    # Types
    "StepType",
    # Context & Result
    "StepContext",
    "StepResult",
    # Step
    "Step",
    # Spec
    "StepSpec",
    "PipelineSpec",
    # Registry
    "StepRegistry",
    "StepMetadata",
    "register_step",
    "get_registry",
    # Validation
    "PipelineValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_pipeline",
    # Runtime
    "RuntimeContext",
    "ExecutionMode",
    "FailureMode",
    "TokenBudget",
    "CostBudget",
    "RBACScope",
    # Executor
    "PipelineExecutor",
    "PipelineResult",
    "StepExecutionResult",
    "execute_pipeline",
    # Async Executor
    "AsyncPipelineExecutor",
    "DryRunExecutor",
    "BatchResult",
    "execute_pipeline_async",
    "execute_batch",
    # Errors
    "RAGOSError",
    "StepError",
    "PipelineError",
    "ProviderError",
    "BudgetExceededError",
    "ConfigurationError",
    "ValidationError",
    "ErrorCode",
    "RetryPolicy",
]
