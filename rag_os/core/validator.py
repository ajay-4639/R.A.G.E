"""Pipeline validator for checking pipeline specifications."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.spec import PipelineSpec, StepSpec
from rag_os.core.registry import StepRegistry, get_registry


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Pipeline cannot run
    WARNING = "warning"  # Pipeline can run but may have issues
    INFO = "info"  # Informational message


@dataclass
class ValidationIssue:
    """A single validation issue found in a pipeline."""

    message: str
    severity: ValidationSeverity
    step_id: str | None = None
    field: str | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        location = f" in step '{self.step_id}'" if self.step_id else ""
        field_info = f" (field: {self.field})" if self.field else ""
        suggestion = f" Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"{prefix}{location}{field_info}: {self.message}{suggestion}"


@dataclass
class ValidationResult:
    """Result of pipeline validation."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        """Get only info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def add_error(
        self,
        message: str,
        step_id: str | None = None,
        field: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add an error issue."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.ERROR,
                step_id=step_id,
                field=field,
                suggestion=suggestion,
            )
        )
        self.valid = False

    def add_warning(
        self,
        message: str,
        step_id: str | None = None,
        field: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a warning issue."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.WARNING,
                step_id=step_id,
                field=field,
                suggestion=suggestion,
            )
        )

    def add_info(
        self,
        message: str,
        step_id: str | None = None,
        field: str | None = None,
    ) -> None:
        """Add an info issue."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.INFO,
                step_id=step_id,
                field=field,
            )
        )

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"Pipeline validation: {status}"]
        lines.append(f"  Errors: {len(self.errors)}, Warnings: {len(self.warnings)}, Info: {len(self.info)}")
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


class PipelineValidator:
    """
    Validator for pipeline specifications.

    Validates that a pipeline spec is correct and can be executed.
    Checks include:
    - All steps exist in the registry
    - Input/output schema compatibility between steps
    - No circular dependencies
    - Config schema validation
    - Provider compatibility
    """

    def __init__(self, registry: StepRegistry | None = None) -> None:
        """
        Initialize the validator.

        Args:
            registry: Step registry to use. If None, uses the global registry.
        """
        self.registry = registry or get_registry()

    def validate(self, spec: PipelineSpec) -> ValidationResult:
        """
        Validate a pipeline specification.

        Args:
            spec: The pipeline specification to validate

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(valid=True)

        # Run all validation checks
        self._validate_steps_exist(spec, result)
        self._validate_step_configs(spec, result)
        self._validate_dependencies(spec, result)
        self._validate_schema_compatibility(spec, result)
        self._validate_fallbacks(spec, result)
        self._validate_pipeline_structure(spec, result)

        return result

    def _validate_steps_exist(self, spec: PipelineSpec, result: ValidationResult) -> None:
        """Check that all steps in the spec exist in the registry."""
        for step in spec.steps:
            if not self.registry.has(step.step_class):
                result.add_error(
                    message=f"Step class '{step.step_class}' is not registered",
                    step_id=step.step_id,
                    field="step_class",
                    suggestion=f"Register the step or use one of: {self.registry.get_by_type(step.step_type)}",
                )

    def _validate_step_configs(self, spec: PipelineSpec, result: ValidationResult) -> None:
        """Validate step configurations against their schemas."""
        for step in spec.steps:
            step_class = self.registry.get(step.step_class)
            if step_class is None:
                continue  # Already reported as missing

            # Create a temporary instance to get the config schema
            try:
                temp_step = step_class(
                    step_id=step.step_id,
                    step_type=step.step_type,
                    config=step.config,
                )
                errors = temp_step.validate_config()
                for error in errors:
                    result.add_error(
                        message=error,
                        step_id=step.step_id,
                        field="config",
                    )
            except Exception as e:
                result.add_error(
                    message=f"Failed to validate config: {e}",
                    step_id=step.step_id,
                    field="config",
                )

    def _validate_dependencies(self, spec: PipelineSpec, result: ValidationResult) -> None:
        """Check for circular dependencies and validate dependency order."""
        step_ids = {s.step_id for s in spec.steps}
        step_order = {s.step_id: i for i, s in enumerate(spec.steps)}

        for step in spec.steps:
            for dep in step.dependencies:
                # Check dependency exists (already validated in PipelineSpec, but double-check)
                if dep not in step_ids:
                    result.add_error(
                        message=f"Dependency '{dep}' does not exist",
                        step_id=step.step_id,
                        field="dependencies",
                    )
                    continue

                # Check dependency comes before this step
                if step_order[dep] >= step_order[step.step_id]:
                    result.add_error(
                        message=f"Dependency '{dep}' must come before step '{step.step_id}'",
                        step_id=step.step_id,
                        field="dependencies",
                        suggestion="Reorder steps so dependencies come first",
                    )

        # Check for circular dependencies
        circular = self._detect_circular_dependencies(spec)
        if circular:
            result.add_error(
                message=f"Circular dependency detected: {' -> '.join(circular)}",
                suggestion="Remove the circular dependency",
            )

    def _detect_circular_dependencies(self, spec: PipelineSpec) -> list[str] | None:
        """Detect circular dependencies using DFS."""
        # Build adjacency list
        graph: dict[str, list[str]] = {s.step_id: s.dependencies.copy() for s in spec.steps}

        # DFS with cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {s.step_id: WHITE for s in spec.steps}
        parent: dict[str, str | None] = {s.step_id: None for s in spec.steps}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for neighbor in graph[node]:
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found cycle, reconstruct path
                    cycle = [neighbor, node]
                    current = node
                    while parent[current] and parent[current] != neighbor:
                        current = parent[current]
                        cycle.append(current)
                    cycle.append(neighbor)
                    return list(reversed(cycle))
                elif color[neighbor] == WHITE:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for step_id in graph:
            if color[step_id] == WHITE:
                cycle = dfs(step_id)
                if cycle:
                    return cycle

        return None

    def _validate_schema_compatibility(self, spec: PipelineSpec, result: ValidationResult) -> None:
        """Check that step input/output schemas are compatible."""
        enabled_steps = spec.get_enabled_steps()

        for i, step in enumerate(enabled_steps[1:], 1):
            # Get previous step (or first dependency)
            prev_step = enabled_steps[i - 1]
            if step.dependencies:
                # Use the first dependency as the primary input source
                dep_id = step.dependencies[0]
                dep_spec = spec.get_step(dep_id)
                if dep_spec:
                    prev_step = dep_spec

            # Get step classes
            current_class = self.registry.get(step.step_class)
            prev_class = self.registry.get(prev_step.step_class)

            if not current_class or not prev_class:
                continue  # Already reported as missing

            # Create temporary instances
            try:
                current_instance = current_class(
                    step_id=step.step_id,
                    step_type=step.step_type,
                    config=step.config,
                )
                prev_instance = prev_class(
                    step_id=prev_step.step_id,
                    step_type=prev_step.step_type,
                    config=prev_step.config,
                )

                # Check type compatibility (simplified check)
                input_type = current_instance.input_schema.get("type")
                output_type = prev_instance.output_schema.get("type")

                if input_type and output_type and input_type != output_type:
                    # This is a warning, not an error, as schemas might still be compatible
                    result.add_warning(
                        message=f"Schema type mismatch: expects '{input_type}' but receives '{output_type}' from '{prev_step.step_id}'",
                        step_id=step.step_id,
                        field="input_schema",
                        suggestion="Verify the data transformation between steps",
                    )

            except Exception as e:
                result.add_warning(
                    message=f"Could not validate schema compatibility: {e}",
                    step_id=step.step_id,
                )

    def _validate_fallbacks(self, spec: PipelineSpec, result: ValidationResult) -> None:
        """Validate fallback step configurations."""
        for step in spec.steps:
            if not step.fallback_step:
                continue

            fallback = spec.get_step(step.fallback_step)
            if not fallback:
                # Already validated in PipelineSpec
                continue

            # Fallback should be of the same type
            if fallback.step_type != step.step_type:
                result.add_warning(
                    message=f"Fallback step '{step.fallback_step}' is of different type ({fallback.step_type}) than main step ({step.step_type})",
                    step_id=step.step_id,
                    field="fallback_step",
                    suggestion="Use a fallback step of the same type",
                )

            # Fallback should ideally be disabled (only used as fallback)
            if fallback.enabled:
                result.add_info(
                    message=f"Fallback step '{step.fallback_step}' is enabled and will run in the normal pipeline flow",
                    step_id=step.step_id,
                    field="fallback_step",
                )

    def _validate_pipeline_structure(self, spec: PipelineSpec, result: ValidationResult) -> None:
        """Validate overall pipeline structure."""
        enabled_steps = spec.get_enabled_steps()

        if not enabled_steps:
            result.add_error(
                message="Pipeline has no enabled steps",
                suggestion="Enable at least one step",
            )
            return

        # Check for recommended step types
        step_types = {s.step_type for s in enabled_steps}

        # A typical RAG pipeline should have retrieval
        if StepType.RETRIEVAL not in step_types:
            result.add_info(
                message="Pipeline does not include a retrieval step",
            )

        # Check first step type
        first_step = enabled_steps[0]
        if first_step.step_type not in (StepType.INGESTION, StepType.RETRIEVAL):
            result.add_info(
                message=f"First step is '{first_step.step_type}', typically pipelines start with ingestion or retrieval",
                step_id=first_step.step_id,
            )


def validate_pipeline(
    spec: PipelineSpec,
    registry: StepRegistry | None = None,
) -> ValidationResult:
    """
    Convenience function to validate a pipeline specification.

    Args:
        spec: The pipeline specification to validate
        registry: Optional step registry (uses global if not provided)

    Returns:
        ValidationResult with all issues found
    """
    validator = PipelineValidator(registry)
    return validator.validate(spec)
