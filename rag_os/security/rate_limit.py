"""Rate limiting for RAG OS."""

import time
import threading
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_second: Maximum requests per second
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size
        per_user: Whether to rate limit per user
    """
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    burst_size: int = 20
    per_user: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit check.

    Attributes:
        allowed: Whether request is allowed
        remaining: Remaining requests in window
        retry_after: Seconds until next request allowed
        limit: The limit that was checked
    """
    allowed: bool = True
    remaining: int = 0
    retry_after: float = 0.0
    limit: str = ""


class TokenBucket:
    """Token bucket algorithm for rate limiting.

    Provides smooth rate limiting with burst support.
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
    ):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill bucket based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate,
        )
        self._last_update = now

    @property
    def available(self) -> float:
        """Get available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            return (tokens - self._tokens) / self.rate


class SlidingWindowCounter:
    """Sliding window counter for rate limiting.

    Provides accurate rate limiting over time windows.
    """

    def __init__(self, window_seconds: float, limit: int):
        """Initialize sliding window.

        Args:
            window_seconds: Window duration in seconds
            limit: Maximum requests in window
        """
        self.window_seconds = window_seconds
        self.limit = limit
        self._requests: list[float] = []
        self._lock = threading.Lock()

    def record(self) -> bool:
        """Record a request and check if allowed.

        Returns:
            True if request is allowed
        """
        now = time.monotonic()

        with self._lock:
            # Remove old requests
            cutoff = now - self.window_seconds
            self._requests = [t for t in self._requests if t > cutoff]

            # Check limit
            if len(self._requests) >= self.limit:
                return False

            # Record new request
            self._requests.append(now)
            return True

    @property
    def count(self) -> int:
        """Get current request count in window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            return sum(1 for t in self._requests if t > cutoff)

    @property
    def remaining(self) -> int:
        """Get remaining requests in window."""
        return max(0, self.limit - self.count)

    def retry_after(self) -> float:
        """Calculate time until next request allowed.

        Returns:
            Seconds to wait
        """
        if self.count < self.limit:
            return 0.0

        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Find oldest request in window
            old_requests = [t for t in self._requests if t > cutoff]
            if old_requests:
                oldest = min(old_requests)
                return (oldest + self.window_seconds) - now

        return 0.0


class RateLimiter:
    """Rate limiter for RAG OS.

    Combines token bucket and sliding window for flexible rate limiting.

    Usage:
        limiter = RateLimiter(config)
        result = limiter.check("user_123")
        if not result.allowed:
            raise RateLimitExceeded(retry_after=result.retry_after)
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, TokenBucket] = {}
        self._windows: dict[str, SlidingWindowCounter] = {}
        self._lock = threading.Lock()

    def check(self, key: str = "default") -> RateLimitResult:
        """Check if request is allowed.

        Args:
            key: Rate limit key (e.g., user ID)

        Returns:
            RateLimitResult
        """
        bucket = self._get_bucket(key)
        window = self._get_window(key)

        # Check token bucket (burst)
        if not bucket.acquire():
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=bucket.wait_time(),
                limit="burst",
            )

        # Check sliding window (sustained rate)
        if not window.record():
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=window.retry_after(),
                limit="rate",
            )

        return RateLimitResult(
            allowed=True,
            remaining=window.remaining,
            retry_after=0.0,
        )

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for key."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    rate=self.config.requests_per_second,
                    capacity=self.config.burst_size,
                )
            return self._buckets[key]

    def _get_window(self, key: str) -> SlidingWindowCounter:
        """Get or create sliding window for key."""
        with self._lock:
            if key not in self._windows:
                self._windows[key] = SlidingWindowCounter(
                    window_seconds=60.0,
                    limit=int(self.config.requests_per_minute),
                )
            return self._windows[key]

    def reset(self, key: str | None = None) -> None:
        """Reset rate limits.

        Args:
            key: Key to reset (None = all)
        """
        with self._lock:
            if key is None:
                self._buckets.clear()
                self._windows.clear()
            else:
                self._buckets.pop(key, None)
                self._windows.pop(key, None)

    def get_status(self, key: str = "default") -> dict[str, Any]:
        """Get rate limit status for a key.

        Args:
            key: Rate limit key

        Returns:
            Status dictionary
        """
        bucket = self._get_bucket(key)
        window = self._get_window(key)

        return {
            "key": key,
            "bucket_available": bucket.available,
            "bucket_capacity": bucket.capacity,
            "window_count": window.count,
            "window_limit": window.limit,
            "window_remaining": window.remaining,
        }
