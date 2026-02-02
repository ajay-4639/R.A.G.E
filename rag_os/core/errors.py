"""Exception hierarchy and error handling for RAG OS."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ErrorCode(str, Enum):
    """Error codes for categorizing errors."""

    # General errors
    UNKNOWN = "UNKNOWN"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

    # Step errors
    STEP_NOT_FOUND = "STEP_NOT_FOUND"
    STEP_EXECUTION_ERROR = "STEP_EXECUTION_ERROR"
    STEP_TIMEOUT = "STEP_TIMEOUT"
    STEP_INPUT_ERROR = "STEP_INPUT_ERROR"
    STEP_OUTPUT_ERROR = "STEP_OUTPUT_ERROR"

    # Pipeline errors
    PIPELINE_NOT_FOUND = "PIPELINE_NOT_FOUND"
    PIPELINE_VALIDATION_ERROR = "PIPELINE_VALIDATION_ERROR"
    PIPELINE_EXECUTION_ERROR = "PIPELINE_EXECUTION_ERROR"

    # Resource errors
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"

    # External service errors
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"

    # Data errors
    DATA_ERROR = "DATA_ERROR"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"


class RAGOSError(Exception):
    """Base exception for all RAG OS errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error": self.__class__.__name__,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class StepError(RAGOSError):
    """Error during step execution."""

    def __init__(
        self,
        message: str,
        step_id: str,
        step_type: str | None = None,
        code: ErrorCode = ErrorCode.STEP_EXECUTION_ERROR,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message, code, details, cause)
        self.step_id = step_id
        self.step_type = step_type
        self.recoverable = recoverable

    def __str__(self) -> str:
        return f"[{self.code.value}] Step '{self.step_id}': {self.message}"

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "step_id": self.step_id,
            "step_type": self.step_type,
            "recoverable": self.recoverable,
        })
        return result


class StepNotFoundError(StepError):
    """Step class not found in registry."""

    def __init__(self, step_id: str, step_class: str) -> None:
        super().__init__(
            message=f"Step class '{step_class}' not found in registry",
            step_id=step_id,
            code=ErrorCode.STEP_NOT_FOUND,
            details={"step_class": step_class},
            recoverable=False,
        )


class StepInputError(StepError):
    """Error in step input validation."""

    def __init__(self, step_id: str, errors: list[str]) -> None:
        super().__init__(
            message=f"Input validation failed: {'; '.join(errors)}",
            step_id=step_id,
            code=ErrorCode.STEP_INPUT_ERROR,
            details={"validation_errors": errors},
            recoverable=False,
        )


class StepOutputError(StepError):
    """Error in step output validation."""

    def __init__(self, step_id: str, errors: list[str]) -> None:
        super().__init__(
            message=f"Output validation failed: {'; '.join(errors)}",
            step_id=step_id,
            code=ErrorCode.STEP_OUTPUT_ERROR,
            details={"validation_errors": errors},
            recoverable=False,
        )


class StepTimeoutError(StepError):
    """Step execution timed out."""

    def __init__(self, step_id: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Step timed out after {timeout_seconds}s",
            step_id=step_id,
            code=ErrorCode.STEP_TIMEOUT,
            details={"timeout_seconds": timeout_seconds},
            recoverable=True,
        )


class PipelineError(RAGOSError):
    """Error during pipeline execution."""

    def __init__(
        self,
        message: str,
        pipeline_name: str | None = None,
        pipeline_version: str | None = None,
        code: ErrorCode = ErrorCode.PIPELINE_EXECUTION_ERROR,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
        failed_steps: list[str] | None = None,
    ) -> None:
        super().__init__(message, code, details, cause)
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.failed_steps = failed_steps or []

    def __str__(self) -> str:
        pipeline_info = ""
        if self.pipeline_name:
            pipeline_info = f" (pipeline: {self.pipeline_name}"
            if self.pipeline_version:
                pipeline_info += f" v{self.pipeline_version}"
            pipeline_info += ")"
        return f"[{self.code.value}]{pipeline_info}: {self.message}"


class BudgetExceededError(RAGOSError):
    """Budget (tokens or cost) exceeded."""

    def __init__(
        self,
        message: str,
        budget_type: str,
        limit: float,
        used: float,
    ) -> None:
        super().__init__(
            message,
            code=ErrorCode.BUDGET_EXCEEDED,
            details={
                "budget_type": budget_type,
                "limit": limit,
                "used": used,
            },
        )
        self.budget_type = budget_type
        self.limit = limit
        self.used = used


class ProviderError(RAGOSError):
    """Error from external provider (LLM, embedding, etc.)."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            code=ErrorCode.PROVIDER_ERROR,
            details=details,
            cause=cause,
        )
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "provider": self.provider,
            "status_code": self.status_code,
            "retryable": self.retryable,
        })
        return result


class RateLimitedError(ProviderError):
    """Rate limited by provider."""

    def __init__(
        self,
        provider: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(
            message=f"Rate limited by {provider}",
            provider=provider,
            status_code=429,
            retryable=True,
            details={"retry_after_seconds": retry_after_seconds},
        )
        self.retry_after_seconds = retry_after_seconds


class ConfigurationError(RAGOSError):
    """Configuration error."""

    def __init__(self, message: str, config_key: str | None = None) -> None:
        super().__init__(
            message,
            code=ErrorCode.CONFIGURATION_ERROR,
            details={"config_key": config_key} if config_key else {},
        )
        self.config_key = config_key


class ValidationError(RAGOSError):
    """Validation error."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(
            message,
            code=ErrorCode.VALIDATION_ERROR,
            details={"errors": errors or []},
        )
        self.errors = errors or []


@dataclass
class RetryPolicy:
    """Policy for retrying failed operations."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    retry_on_errors: list[type[Exception]] = field(default_factory=list)
    retry_on_codes: list[ErrorCode] = field(default_factory=list)

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if the operation should be retried."""
        if attempt >= self.max_retries:
            return False

        # Check error type
        if self.retry_on_errors:
            if any(isinstance(error, err_type) for err_type in self.retry_on_errors):
                return True

        # Check error code for RAGOSError
        if isinstance(error, RAGOSError) and self.retry_on_codes:
            if error.code in self.retry_on_codes:
                return True

        # Check if error is explicitly recoverable
        if isinstance(error, StepError) and error.recoverable:
            return True

        # Check if provider error is retryable
        if isinstance(error, ProviderError) and error.retryable:
            return True

        return False

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        delay = self.initial_delay_seconds * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_seconds)


def create_default_retry_policy() -> RetryPolicy:
    """Create a default retry policy with common settings."""
    return RetryPolicy(
        max_retries=3,
        initial_delay_seconds=1.0,
        backoff_multiplier=2.0,
        retry_on_errors=[TimeoutError, ConnectionError],
        retry_on_codes=[ErrorCode.STEP_TIMEOUT, ErrorCode.RATE_LIMITED, ErrorCode.NETWORK_ERROR],
    )
