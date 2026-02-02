"""Secret management for RAG OS."""

import re
import os
from dataclasses import dataclass, field
from typing import Any
import threading


@dataclass
class Secret:
    """A stored secret.

    Attributes:
        name: Secret name
        value: Secret value (stored encrypted in production)
        source: Where the secret came from
    """
    name: str
    value: str
    source: str = "manual"

    def __repr__(self) -> str:
        """Safe representation that doesn't expose value."""
        return f"Secret(name={self.name!r}, source={self.source!r})"


class SecretManager:
    """Manager for secrets and sensitive configuration.

    Provides secure storage and retrieval of secrets like API keys.

    Usage:
        manager = SecretManager()
        manager.set("OPENAI_API_KEY", "sk-...")
        key = manager.get("OPENAI_API_KEY")
    """

    def __init__(
        self,
        load_from_env: bool = True,
        env_prefix: str = "RAG_OS_",
    ):
        """Initialize secret manager.

        Args:
            load_from_env: Whether to load from environment
            env_prefix: Prefix for environment variables
        """
        self._secrets: dict[str, Secret] = {}
        self._lock = threading.Lock()
        self._env_prefix = env_prefix

        if load_from_env:
            self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load secrets from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                name = key[len(self._env_prefix):]
                self._secrets[name] = Secret(
                    name=name,
                    value=value,
                    source="environment",
                )

    def set(self, name: str, value: str, source: str = "manual") -> None:
        """Set a secret.

        Args:
            name: Secret name
            value: Secret value
            source: Source of the secret
        """
        with self._lock:
            self._secrets[name] = Secret(name=name, value=value, source=source)

    def get(self, name: str, default: str | None = None) -> str | None:
        """Get a secret.

        Args:
            name: Secret name
            default: Default value if not found

        Returns:
            Secret value or default
        """
        with self._lock:
            secret = self._secrets.get(name)
            if secret:
                return secret.value

            # Try environment as fallback
            env_value = os.environ.get(f"{self._env_prefix}{name}")
            if env_value:
                return env_value

            return default

    def delete(self, name: str) -> bool:
        """Delete a secret.

        Args:
            name: Secret name

        Returns:
            True if deleted
        """
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                return True
            return False

    def exists(self, name: str) -> bool:
        """Check if a secret exists.

        Args:
            name: Secret name

        Returns:
            True if exists
        """
        with self._lock:
            if name in self._secrets:
                return True
            return f"{self._env_prefix}{name}" in os.environ

    def list_secrets(self) -> list[str]:
        """Get list of secret names (not values)."""
        with self._lock:
            return list(self._secrets.keys())

    def get_source(self, name: str) -> str | None:
        """Get the source of a secret.

        Args:
            name: Secret name

        Returns:
            Source or None
        """
        with self._lock:
            secret = self._secrets.get(name)
            return secret.source if secret else None


# Common patterns for secrets
SECRET_PATTERNS = [
    # API keys
    (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key"),
    (r"sk-ant-[a-zA-Z0-9-]{32,}", "Anthropic API key"),
    (r"AIza[0-9A-Za-z-_]{35}", "Google API key"),
    (r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}", "Slack bot token"),

    # AWS
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"[a-zA-Z0-9/+=]{40}", "AWS Secret Key (potential)"),

    # Generic patterns
    (r"['\"][a-zA-Z0-9_-]{20,}['\"]", "Potential API key"),
    (r"password['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", "Password assignment"),
    (r"api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", "API key assignment"),
    (r"secret['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", "Secret assignment"),
    (r"token['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", "Token assignment"),
]


def mask_secrets(
    text: str,
    mask_char: str = "*",
    visible_chars: int = 4,
    patterns: list[tuple[str, str]] | None = None,
) -> str:
    """Mask secrets in text.

    Replaces detected secrets with masked versions.

    Args:
        text: Text to mask
        mask_char: Character to use for masking
        visible_chars: Number of chars to leave visible
        patterns: Custom patterns to detect

    Returns:
        Text with secrets masked
    """
    if not text:
        return text

    patterns = patterns or SECRET_PATTERNS

    masked_text = text

    for pattern, _ in patterns:
        def mask_match(match: re.Match) -> str:
            value = match.group(0)
            if len(value) <= visible_chars * 2:
                return mask_char * len(value)
            return value[:visible_chars] + mask_char * (len(value) - visible_chars * 2) + value[-visible_chars:]

        masked_text = re.sub(pattern, mask_match, masked_text, flags=re.IGNORECASE)

    return masked_text


def detect_secrets(text: str, patterns: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Detect potential secrets in text.

    Args:
        text: Text to scan
        patterns: Custom patterns to detect

    Returns:
        List of detected secrets with positions
    """
    if not text:
        return []

    patterns = patterns or SECRET_PATTERNS
    detected = []

    for pattern, description in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            detected.append({
                "type": description,
                "pattern": pattern,
                "start": match.start(),
                "end": match.end(),
                "masked": mask_secrets(match.group(0)),
            })

    return detected


def is_secret_exposed(value: str, patterns: list[tuple[str, str]] | None = None) -> bool:
    """Check if a value looks like an exposed secret.

    Args:
        value: Value to check
        patterns: Custom patterns

    Returns:
        True if value matches secret patterns
    """
    patterns = patterns or SECRET_PATTERNS

    for pattern, _ in patterns:
        if re.fullmatch(pattern, value):
            return True

    return False
