"""Unit tests for error handling."""

import pytest

from rag_os.core.errors import (
    ErrorCode,
    RAGOSError,
    StepError,
    StepNotFoundError,
    StepInputError,
    StepOutputError,
    StepTimeoutError,
    PipelineError,
    BudgetExceededError,
    ProviderError,
    RateLimitedError,
    ConfigurationError,
    ValidationError,
    RetryPolicy,
    create_default_retry_policy,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_all_codes_exist(self):
        """All expected error codes exist."""
        codes = [c.value for c in ErrorCode]
        assert "UNKNOWN" in codes
        assert "STEP_NOT_FOUND" in codes
        assert "BUDGET_EXCEEDED" in codes
        assert "PROVIDER_ERROR" in codes


class TestRAGOSError:
    """Tests for base RAGOSError."""

    def test_basic_error(self):
        """Can create basic error."""
        error = RAGOSError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.code == ErrorCode.UNKNOWN
        assert error.details == {}
        assert error.cause is None

    def test_error_with_details(self):
        """Error can have details."""
        error = RAGOSError(
            "Validation failed",
            code=ErrorCode.VALIDATION_ERROR,
            details={"field": "name"},
        )

        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.details["field"] == "name"

    def test_error_with_cause(self):
        """Error can wrap another exception."""
        original = ValueError("original error")
        error = RAGOSError("Wrapped error", cause=original)

        assert error.cause is original

    def test_error_str(self):
        """Error has useful string representation."""
        error = RAGOSError("Test error", code=ErrorCode.DATA_ERROR)

        assert "[DATA_ERROR]" in str(error)
        assert "Test error" in str(error)

    def test_error_to_dict(self):
        """Error can be converted to dict."""
        error = RAGOSError(
            "Test",
            code=ErrorCode.CONFIGURATION_ERROR,
            details={"key": "value"},
        )

        d = error.to_dict()
        assert d["error"] == "RAGOSError"
        assert d["code"] == "CONFIGURATION_ERROR"
        assert d["message"] == "Test"
        assert d["details"]["key"] == "value"


class TestStepError:
    """Tests for StepError."""

    def test_basic_step_error(self):
        """Can create step error."""
        error = StepError("Failed", step_id="step1")

        assert error.step_id == "step1"
        assert error.code == ErrorCode.STEP_EXECUTION_ERROR
        assert not error.recoverable

    def test_step_error_str(self):
        """Step error includes step ID."""
        error = StepError("Failed", step_id="my_step")

        assert "my_step" in str(error)
        assert "Failed" in str(error)

    def test_step_error_recoverable(self):
        """Step error can be recoverable."""
        error = StepError("Timeout", step_id="s1", recoverable=True)

        assert error.recoverable

    def test_step_error_to_dict(self):
        """Step error dict includes step info."""
        error = StepError(
            "Error",
            step_id="s1",
            step_type="chunking",
            recoverable=True,
        )

        d = error.to_dict()
        assert d["step_id"] == "s1"
        assert d["step_type"] == "chunking"
        assert d["recoverable"] is True


class TestStepNotFoundError:
    """Tests for StepNotFoundError."""

    def test_step_not_found(self):
        """Creates proper not found error."""
        error = StepNotFoundError(step_id="s1", step_class="MissingStep")

        assert error.code == ErrorCode.STEP_NOT_FOUND
        assert "MissingStep" in error.message
        assert error.details["step_class"] == "MissingStep"
        assert not error.recoverable


class TestStepInputError:
    """Tests for StepInputError."""

    def test_input_error(self):
        """Creates proper input validation error."""
        error = StepInputError(step_id="s1", errors=["Missing field 'x'", "Invalid type"])

        assert error.code == ErrorCode.STEP_INPUT_ERROR
        assert "Missing field" in error.message
        assert len(error.details["validation_errors"]) == 2


class TestStepOutputError:
    """Tests for StepOutputError."""

    def test_output_error(self):
        """Creates proper output validation error."""
        error = StepOutputError(step_id="s1", errors=["Missing required field"])

        assert error.code == ErrorCode.STEP_OUTPUT_ERROR
        assert "validation" in error.message.lower()


class TestStepTimeoutError:
    """Tests for StepTimeoutError."""

    def test_timeout_error(self):
        """Creates proper timeout error."""
        error = StepTimeoutError(step_id="slow_step", timeout_seconds=30.0)

        assert error.code == ErrorCode.STEP_TIMEOUT
        assert error.recoverable  # Timeouts are recoverable
        assert "30" in error.message
        assert error.details["timeout_seconds"] == 30.0


class TestPipelineError:
    """Tests for PipelineError."""

    def test_basic_pipeline_error(self):
        """Can create pipeline error."""
        error = PipelineError("Pipeline failed")

        assert error.code == ErrorCode.PIPELINE_EXECUTION_ERROR
        assert error.failed_steps == []

    def test_pipeline_error_with_context(self):
        """Pipeline error includes pipeline info."""
        error = PipelineError(
            "Failed",
            pipeline_name="my-pipeline",
            pipeline_version="1.0.0",
            failed_steps=["step1", "step2"],
        )

        assert error.pipeline_name == "my-pipeline"
        assert error.pipeline_version == "1.0.0"
        assert len(error.failed_steps) == 2
        assert "my-pipeline" in str(error)


class TestBudgetExceededError:
    """Tests for BudgetExceededError."""

    def test_budget_error(self):
        """Creates proper budget error."""
        error = BudgetExceededError(
            message="Token budget exceeded",
            budget_type="tokens",
            limit=1000,
            used=1500,
        )

        assert error.code == ErrorCode.BUDGET_EXCEEDED
        assert error.budget_type == "tokens"
        assert error.limit == 1000
        assert error.used == 1500


class TestProviderError:
    """Tests for ProviderError."""

    def test_basic_provider_error(self):
        """Can create provider error."""
        error = ProviderError(
            "API error",
            provider="openai",
            status_code=500,
        )

        assert error.provider == "openai"
        assert error.status_code == 500
        assert not error.retryable

    def test_retryable_provider_error(self):
        """Provider error can be retryable."""
        error = ProviderError(
            "Temporary failure",
            provider="anthropic",
            retryable=True,
        )

        assert error.retryable

    def test_provider_error_to_dict(self):
        """Provider error dict includes provider info."""
        error = ProviderError(
            "Error",
            provider="cohere",
            status_code=429,
            retryable=True,
        )

        d = error.to_dict()
        assert d["provider"] == "cohere"
        assert d["status_code"] == 429
        assert d["retryable"] is True


class TestRateLimitedError:
    """Tests for RateLimitedError."""

    def test_rate_limit_error(self):
        """Creates proper rate limit error."""
        error = RateLimitedError(provider="openai", retry_after_seconds=60.0)

        assert error.code == ErrorCode.PROVIDER_ERROR  # Inherits from ProviderError
        assert error.status_code == 429
        assert error.retryable
        assert error.retry_after_seconds == 60.0


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_config_error(self):
        """Creates proper config error."""
        error = ConfigurationError("Missing API key", config_key="openai_api_key")

        assert error.code == ErrorCode.CONFIGURATION_ERROR
        assert error.config_key == "openai_api_key"


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error(self):
        """Creates proper validation error."""
        error = ValidationError(
            "Validation failed",
            errors=["Field 'x' is required", "Field 'y' must be positive"],
        )

        assert error.code == ErrorCode.VALIDATION_ERROR
        assert len(error.errors) == 2


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_policy(self):
        """Default policy has sensible values."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.initial_delay_seconds == 1.0
        assert policy.backoff_multiplier == 2.0

    def test_should_retry_under_max(self):
        """Should retry if under max retries."""
        policy = RetryPolicy(max_retries=3, retry_on_errors=[ValueError])
        error = ValueError("test")

        assert policy.should_retry(error, attempt=0)
        assert policy.should_retry(error, attempt=1)
        assert policy.should_retry(error, attempt=2)
        assert not policy.should_retry(error, attempt=3)

    def test_should_retry_wrong_error_type(self):
        """Should not retry for wrong error type."""
        policy = RetryPolicy(retry_on_errors=[ValueError])
        error = TypeError("test")

        assert not policy.should_retry(error, attempt=0)

    def test_should_retry_error_code(self):
        """Should retry for matching error code."""
        policy = RetryPolicy(retry_on_codes=[ErrorCode.STEP_TIMEOUT])
        error = StepTimeoutError(step_id="s1", timeout_seconds=30)

        assert policy.should_retry(error, attempt=0)

    def test_should_retry_recoverable(self):
        """Should retry recoverable errors."""
        policy = RetryPolicy()
        error = StepError("test", step_id="s1", recoverable=True)

        assert policy.should_retry(error, attempt=0)

    def test_should_retry_provider_retryable(self):
        """Should retry retryable provider errors."""
        policy = RetryPolicy()
        error = ProviderError("test", provider="openai", retryable=True)

        assert policy.should_retry(error, attempt=0)

    def test_get_delay_exponential(self):
        """Delay increases exponentially."""
        policy = RetryPolicy(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=60.0,
        )

        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

    def test_get_delay_capped(self):
        """Delay is capped at max."""
        policy = RetryPolicy(
            initial_delay_seconds=10.0,
            backoff_multiplier=10.0,
            max_delay_seconds=30.0,
        )

        assert policy.get_delay(0) == 10.0
        assert policy.get_delay(1) == 30.0  # Capped at max
        assert policy.get_delay(2) == 30.0  # Still capped


class TestCreateDefaultRetryPolicy:
    """Tests for create_default_retry_policy."""

    def test_default_policy(self):
        """Default policy has common settings."""
        policy = create_default_retry_policy()

        assert policy.max_retries == 3
        assert TimeoutError in policy.retry_on_errors
        assert ConnectionError in policy.retry_on_errors
        assert ErrorCode.STEP_TIMEOUT in policy.retry_on_codes
        assert ErrorCode.RATE_LIMITED in policy.retry_on_codes
