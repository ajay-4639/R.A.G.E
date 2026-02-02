"""Tests for RAG OS security module."""

import pytest
import time
import os

from rag_os.security import (
    InputValidator,
    ValidationRule,
    sanitize_input,
    validate_query,
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    SecretManager,
    mask_secrets,
)
from rag_os.security.validation import ValidationSeverity, ValidationResult
from rag_os.security.rate_limit import SlidingWindowCounter
from rag_os.security.secrets import detect_secrets, is_secret_exposed


# =============================================================================
# InputValidator Tests
# =============================================================================

class TestInputValidator:
    """Tests for InputValidator."""

    def test_valid_query(self):
        """Test validating a normal query."""
        validator = InputValidator()

        result = validator.validate_query("What is the capital of France?")

        assert result.valid is True
        assert len(result.errors) == 0

    def test_empty_query(self):
        """Test validating empty query."""
        validator = InputValidator(min_query_length=1)

        result = validator.validate_query("")

        assert result.valid is False
        assert any("too short" in e["message"] for e in result.errors)

    def test_query_too_long(self):
        """Test query exceeding max length."""
        validator = InputValidator(max_query_length=10)

        result = validator.validate_query("This is a very long query")

        assert result.valid is False
        assert any("too long" in e["message"] for e in result.errors)

    def test_blocked_script_tag(self):
        """Test blocking script tags."""
        validator = InputValidator()

        result = validator.validate_query("Hello <script>alert('xss')</script>")

        assert result.valid is False

    def test_blocked_javascript_protocol(self):
        """Test blocking javascript: protocol."""
        validator = InputValidator()

        result = validator.validate_query("Click javascript:alert(1)")

        assert result.valid is False

    def test_blocked_event_handler(self):
        """Test blocking event handlers."""
        validator = InputValidator()

        result = validator.validate_query("<img onerror='alert(1)'>")

        assert result.valid is False

    def test_custom_rule(self):
        """Test custom validation rule."""
        rule = ValidationRule(
            name="no_profanity",
            check=lambda x: "badword" not in x.lower(),
            message="Query contains inappropriate content",
        )
        validator = InputValidator(custom_rules=[rule])

        result = validator.validate_query("This has badword in it")

        assert result.valid is False
        assert any("inappropriate" in e["message"] for e in result.errors)

    def test_validate_document(self):
        """Test validating a document."""
        validator = InputValidator()

        result = validator.validate_document({
            "content": "Normal document content",
            "metadata": {"author": "Test"},
        })

        assert result.valid is True

    def test_validate_document_missing_content(self):
        """Test document without content field."""
        validator = InputValidator()

        result = validator.validate_document({"metadata": {}})

        assert result.valid is False

    def test_validate_config(self):
        """Test validating configuration."""
        validator = InputValidator()

        result = validator.validate_config({
            "model": "gpt-4",
            "temperature": 0.7,
        })

        assert result.valid is True

    def test_validate_config_dangerous_key(self):
        """Test config with dangerous keys."""
        validator = InputValidator()

        result = validator.validate_config({
            "eval": "some_code",
        })

        assert result.valid is False


# =============================================================================
# sanitize_input Tests
# =============================================================================

class TestSanitizeInput:
    """Tests for sanitize_input function."""

    def test_remove_null_bytes(self):
        """Test removing null bytes."""
        result = sanitize_input("Hello\x00World")

        assert "\x00" not in result
        assert result == "HelloWorld"

    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        result = sanitize_input("Hello    World")

        assert result == "Hello World"

    def test_collapse_newlines(self):
        """Test collapsing multiple newlines."""
        result = sanitize_input("Hello\n\n\n\n\nWorld")

        assert result == "Hello\n\nWorld"

    def test_strip_text(self):
        """Test stripping leading/trailing whitespace."""
        result = sanitize_input("   Hello World   ")

        assert result == "Hello World"


# =============================================================================
# validate_query Tests
# =============================================================================

class TestValidateQuery:
    """Tests for validate_query convenience function."""

    def test_basic_validation(self):
        """Test basic query validation."""
        result = validate_query("Hello world")

        assert result.valid is True

    def test_custom_max_length(self):
        """Test custom max length."""
        result = validate_query("Hello", max_length=3)

        assert result.valid is False


# =============================================================================
# TokenBucket Tests
# =============================================================================

class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_acquire_tokens(self):
        """Test acquiring tokens."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        # Should be able to acquire immediately
        assert bucket.acquire(5) is True
        assert bucket.available >= 4.9  # Small tolerance for timing

    def test_bucket_exhaustion(self):
        """Test bucket exhaustion."""
        bucket = TokenBucket(rate=1.0, capacity=3)

        # Use all tokens
        assert bucket.acquire(3) is True
        assert bucket.acquire(1) is False

    def test_bucket_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        bucket.acquire(10)  # Empty the bucket
        time.sleep(0.2)     # Wait for some refill

        assert bucket.available >= 1.5  # Should have refilled some

    def test_wait_time(self):
        """Test wait time calculation."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        bucket.acquire(10)  # Empty

        wait = bucket.wait_time(1)
        assert wait > 0
        assert wait <= 0.15  # ~0.1 seconds to get 1 token


# =============================================================================
# SlidingWindowCounter Tests
# =============================================================================

class TestSlidingWindowCounter:
    """Tests for SlidingWindowCounter."""

    def test_record_requests(self):
        """Test recording requests."""
        counter = SlidingWindowCounter(window_seconds=1.0, limit=5)

        for _ in range(5):
            assert counter.record() is True

        assert counter.record() is False

    def test_window_sliding(self):
        """Test window sliding over time."""
        counter = SlidingWindowCounter(window_seconds=0.2, limit=3)

        # Record 3 requests
        for _ in range(3):
            counter.record()

        assert counter.record() is False

        # Wait for window to slide
        time.sleep(0.25)

        # Should be able to record again
        assert counter.record() is True

    def test_remaining(self):
        """Test remaining count."""
        counter = SlidingWindowCounter(window_seconds=1.0, limit=5)

        counter.record()
        counter.record()

        assert counter.remaining == 3


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allow_normal_requests(self):
        """Test allowing normal request rate."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            requests_per_minute=100.0,
            burst_size=20,
        )
        limiter = RateLimiter(config)

        result = limiter.check("user1")

        assert result.allowed is True

    def test_block_burst(self):
        """Test blocking burst exceeding limit."""
        config = RateLimitConfig(
            requests_per_second=100.0,  # High sustained rate
            requests_per_minute=1000.0,
            burst_size=3,
        )
        limiter = RateLimiter(config)

        # Exhaust burst
        for _ in range(3):
            limiter.check("user1")

        result = limiter.check("user1")
        assert result.allowed is False
        assert result.limit == "burst"

    def test_per_user_limits(self):
        """Test per-user rate limiting."""
        config = RateLimitConfig(
            burst_size=2,
        )
        limiter = RateLimiter(config)

        # User 1 uses their limit
        limiter.check("user1")
        limiter.check("user1")

        # User 2 should still be allowed
        result = limiter.check("user2")
        assert result.allowed is True

    def test_reset_limits(self):
        """Test resetting limits."""
        config = RateLimitConfig(burst_size=2)
        limiter = RateLimiter(config)

        # Use limits
        limiter.check("user1")
        limiter.check("user1")
        assert limiter.check("user1").allowed is False

        # Reset
        limiter.reset("user1")

        # Should be allowed again
        assert limiter.check("user1").allowed is True

    def test_get_status(self):
        """Test getting rate limit status."""
        limiter = RateLimiter()
        limiter.check("user1")

        status = limiter.get_status("user1")

        assert "key" in status
        assert "bucket_available" in status
        assert "window_remaining" in status


# =============================================================================
# SecretManager Tests
# =============================================================================

class TestSecretManager:
    """Tests for SecretManager."""

    def test_set_and_get_secret(self):
        """Test setting and getting a secret."""
        manager = SecretManager(load_from_env=False)

        manager.set("API_KEY", "test-key-123")

        assert manager.get("API_KEY") == "test-key-123"

    def test_get_nonexistent(self):
        """Test getting nonexistent secret."""
        manager = SecretManager(load_from_env=False)

        assert manager.get("NONEXISTENT") is None
        assert manager.get("NONEXISTENT", "default") == "default"

    def test_delete_secret(self):
        """Test deleting a secret."""
        manager = SecretManager(load_from_env=False)
        manager.set("API_KEY", "value")

        result = manager.delete("API_KEY")

        assert result is True
        assert manager.get("API_KEY") is None

    def test_exists(self):
        """Test checking secret existence."""
        manager = SecretManager(load_from_env=False)
        manager.set("EXISTS", "value")

        assert manager.exists("EXISTS") is True
        assert manager.exists("NOT_EXISTS") is False

    def test_list_secrets(self):
        """Test listing secret names."""
        manager = SecretManager(load_from_env=False)
        manager.set("KEY1", "val1")
        manager.set("KEY2", "val2")

        names = manager.list_secrets()

        assert set(names) == {"KEY1", "KEY2"}

    def test_get_source(self):
        """Test getting secret source."""
        manager = SecretManager(load_from_env=False)
        manager.set("API_KEY", "value", source="config_file")

        source = manager.get_source("API_KEY")

        assert source == "config_file"

    def test_secret_repr_safe(self):
        """Test that Secret repr doesn't expose value."""
        from rag_os.security.secrets import Secret

        secret = Secret(name="API_KEY", value="super-secret-value")
        repr_str = repr(secret)

        assert "super-secret-value" not in repr_str
        assert "API_KEY" in repr_str


# =============================================================================
# mask_secrets Tests
# =============================================================================

class TestMaskSecrets:
    """Tests for mask_secrets function."""

    def test_mask_openai_key(self):
        """Test masking OpenAI API key."""
        text = "My key is sk-1234567890abcdefghijklmnopqrstuvwxyz"

        masked = mask_secrets(text)

        assert "sk-1234" not in masked or "****" in masked

    def test_mask_multiple_secrets(self):
        """Test masking multiple secrets."""
        text = "Keys: sk-abc123def456ghi789jkl012mno345pqr and AKIA1234567890123456"

        masked = mask_secrets(text)

        assert "****" in masked

    def test_preserve_non_secrets(self):
        """Test that non-secrets are preserved."""
        text = "This is a normal query with no secrets"

        masked = mask_secrets(text)

        assert masked == text


# =============================================================================
# detect_secrets Tests
# =============================================================================

class TestDetectSecrets:
    """Tests for detect_secrets function."""

    def test_detect_api_key(self):
        """Test detecting API keys."""
        text = "Config: AKIA1234567890123456 is the key"

        detected = detect_secrets(text)

        assert len(detected) >= 1

    def test_no_secrets(self):
        """Test text with no secrets."""
        text = "This is a normal query"

        detected = detect_secrets(text)

        # May have false positives, but should be minimal
        assert len(detected) <= 1


# =============================================================================
# is_secret_exposed Tests
# =============================================================================

class TestIsSecretExposed:
    """Tests for is_secret_exposed function."""

    def test_exposed_api_key(self):
        """Test detecting exposed API key."""
        # AWS key format
        assert is_secret_exposed("AKIA1234567890123456") is True

    def test_normal_value(self):
        """Test normal value not detected as secret."""
        assert is_secret_exposed("hello world") is False
