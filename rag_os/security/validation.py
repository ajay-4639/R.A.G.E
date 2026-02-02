"""Input validation for RAG OS."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of validation.

    Attributes:
        valid: Whether validation passed
        issues: List of validation issues
        sanitized: Sanitized value (if applicable)
    """
    valid: bool = True
    issues: list[dict[str, Any]] = field(default_factory=list)
    sanitized: Any = None

    def add_issue(
        self,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        field: str | None = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append({
            "message": message,
            "severity": severity.value,
            "field": field,
        })
        if severity == ValidationSeverity.ERROR:
            self.valid = False

    @property
    def errors(self) -> list[dict[str, Any]]:
        """Get only error-level issues."""
        return [i for i in self.issues if i["severity"] == "error"]


@dataclass
class ValidationRule:
    """A validation rule.

    Attributes:
        name: Rule name
        check: Function that returns True if valid
        message: Error message if invalid
        severity: Issue severity
    """
    name: str
    check: Callable[[Any], bool]
    message: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR


class InputValidator:
    """Validator for user inputs.

    Provides validation for queries, documents, and configurations.

    Usage:
        validator = InputValidator()
        result = validator.validate_query("user query")
        if not result.valid:
            print(result.errors)
    """

    # Default blocked patterns (potential injections)
    DEFAULT_BLOCKED_PATTERNS = [
        r"<script\b[^>]*>",  # Script tags
        r"javascript:",      # JS protocol
        r"on\w+\s*=",        # Event handlers
        r"\{\{.*\}\}",       # Template injection
        r"\$\{.*\}",         # Template literals
    ]

    def __init__(
        self,
        max_query_length: int = 10000,
        min_query_length: int = 1,
        blocked_patterns: list[str] | None = None,
        custom_rules: list[ValidationRule] | None = None,
    ):
        """Initialize validator.

        Args:
            max_query_length: Maximum query length
            min_query_length: Minimum query length
            blocked_patterns: Regex patterns to block
            custom_rules: Additional validation rules
        """
        self.max_query_length = max_query_length
        self.min_query_length = min_query_length
        self.blocked_patterns = blocked_patterns or self.DEFAULT_BLOCKED_PATTERNS
        self.custom_rules = custom_rules or []

        # Compile patterns
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
        ]

    def validate_query(self, query: str) -> ValidationResult:
        """Validate a query string.

        Args:
            query: Query to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # Type check
        if not isinstance(query, str):
            result.add_issue("Query must be a string", field="query")
            return result

        # Length checks
        if len(query) < self.min_query_length:
            result.add_issue(
                f"Query too short (min {self.min_query_length} chars)",
                field="query",
            )

        if len(query) > self.max_query_length:
            result.add_issue(
                f"Query too long (max {self.max_query_length} chars)",
                field="query",
            )

        # Pattern checks
        for pattern in self._compiled_patterns:
            if pattern.search(query):
                result.add_issue(
                    f"Query contains blocked pattern: {pattern.pattern}",
                    field="query",
                )

        # Custom rules
        for rule in self.custom_rules:
            if not rule.check(query):
                result.add_issue(
                    rule.message or f"Failed rule: {rule.name}",
                    severity=rule.severity,
                    field="query",
                )

        # Sanitize
        result.sanitized = sanitize_input(query) if result.valid else query

        return result

    def validate_document(self, document: dict[str, Any]) -> ValidationResult:
        """Validate a document.

        Args:
            document: Document to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # Type check
        if not isinstance(document, dict):
            result.add_issue("Document must be a dictionary", field="document")
            return result

        # Check for required fields
        if "content" not in document and "text" not in document:
            result.add_issue(
                "Document must have 'content' or 'text' field",
                field="document",
            )

        # Validate content if present
        content = document.get("content") or document.get("text", "")
        if content:
            content_result = self.validate_query(content)
            if not content_result.valid:
                for issue in content_result.issues:
                    result.add_issue(
                        issue["message"],
                        severity=ValidationSeverity(issue["severity"]),
                        field="document.content",
                    )

        return result

    def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary.

        Args:
            config: Config to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        if not isinstance(config, dict):
            result.add_issue("Config must be a dictionary", field="config")
            return result

        # Check for dangerous keys
        dangerous_keys = ["__import__", "eval", "exec", "compile", "globals", "locals"]
        for key in config.keys():
            if key in dangerous_keys:
                result.add_issue(
                    f"Config contains potentially dangerous key: {key}",
                    field=f"config.{key}",
                )

        # Recursively check string values
        self._validate_config_values(config, result, "config")

        return result

    def _validate_config_values(
        self,
        obj: Any,
        result: ValidationResult,
        path: str,
    ) -> None:
        """Recursively validate config values."""
        if isinstance(obj, str):
            for pattern in self._compiled_patterns:
                if pattern.search(obj):
                    result.add_issue(
                        f"Config value contains blocked pattern",
                        severity=ValidationSeverity.WARNING,
                        field=path,
                    )
        elif isinstance(obj, dict):
            for key, value in obj.items():
                self._validate_config_values(value, result, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                self._validate_config_values(value, result, f"{path}[{i}]")


def sanitize_input(text: str) -> str:
    """Sanitize text input by removing potentially dangerous content.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters (except common whitespace)
    text = "".join(
        char for char in text
        if char.isprintable() or char in "\n\r\t"
    )

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def validate_query(
    query: str,
    max_length: int = 10000,
    min_length: int = 1,
) -> ValidationResult:
    """Convenience function to validate a query.

    Args:
        query: Query to validate
        max_length: Maximum allowed length
        min_length: Minimum required length

    Returns:
        ValidationResult
    """
    validator = InputValidator(
        max_query_length=max_length,
        min_query_length=min_length,
    )
    return validator.validate_query(query)
