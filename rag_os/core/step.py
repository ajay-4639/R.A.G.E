"""Base Step abstract class - the foundation of all pipeline steps."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_os.core.types import StepType, SchemaType, ConfigType
from rag_os.core.context import StepContext
from rag_os.core.result import StepResult


class StepConfig(BaseModel):
    """Base configuration for all steps."""

    model_config = ConfigDict(extra="allow")


class Step(ABC):
    """
    Abstract base class for all pipeline steps.

    Every step in a RAG pipeline must inherit from this class and implement
    the required methods. Steps are the building blocks of pipelines.

    Attributes:
        step_id: Unique identifier for this step instance
        step_type: The category of this step (e.g., INGESTION, CHUNKING)
        config: Step-specific configuration
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType,
        config: ConfigType | None = None,
    ) -> None:
        """
        Initialize a step.

        Args:
            step_id: Unique identifier for this step instance
            step_type: The type/category of this step
            config: Optional configuration dictionary
        """
        self.step_id = step_id
        self.step_type = step_type
        self.config = config or {}

    @property
    @abstractmethod
    def input_schema(self) -> SchemaType:
        """
        Return the JSON schema for the expected input.

        This schema defines what data structure the step expects
        in the context.data field.
        """
        ...

    @property
    @abstractmethod
    def output_schema(self) -> SchemaType:
        """
        Return the JSON schema for the output.

        This schema defines what data structure the step will
        produce in the StepResult.output field.
        """
        ...

    @property
    def config_schema(self) -> SchemaType:
        """
        Return the JSON schema for step configuration.

        Override this in subclasses to define config validation.
        Default returns an empty object schema (any config allowed).
        """
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @property
    def runtime_requirements(self) -> dict[str, Any]:
        """
        Return runtime requirements for this step.

        This can include things like:
        - required_memory: str (e.g., "2GB")
        - requires_gpu: bool
        - estimated_latency_ms: int
        - external_services: list[str]

        Default returns empty dict (no special requirements).
        """
        return {}

    def validate_input(self, context: StepContext) -> list[str]:
        """
        Validate that the input context matches the expected schema.

        Args:
            context: The input context to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Basic validation - override for more specific checks
        if self.input_schema.get("type") == "object":
            required = self.input_schema.get("required", [])
            if isinstance(context.data, dict):
                for field in required:
                    if field not in context.data:
                        errors.append(f"Missing required field: {field}")
            elif context.data is None and required:
                errors.append("Input data is None but schema requires fields")

        return errors

    def validate_output(self, result: StepResult) -> list[str]:
        """
        Validate that the output result matches the expected schema.

        Args:
            result: The step result to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        if not result.success:
            # Don't validate output of failed steps
            return errors

        # Basic validation - override for more specific checks
        if self.output_schema.get("type") == "object":
            required = self.output_schema.get("required", [])
            if isinstance(result.output, dict):
                for field in required:
                    if field not in result.output:
                        errors.append(f"Missing required output field: {field}")
            elif result.output is None and required:
                errors.append("Output is None but schema requires fields")

        return errors

    def validate_config(self) -> list[str]:
        """
        Validate the step's configuration against its config schema.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        schema = self.config_schema
        required = schema.get("required", [])

        for field in required:
            if field not in self.config:
                errors.append(f"Missing required config field: {field}")

        return errors

    @abstractmethod
    def execute(self, context: StepContext) -> StepResult:
        """
        Execute the step's logic.

        This is the main method that performs the step's work.
        Subclasses must implement this method.

        Args:
            context: The input context containing data and metadata

        Returns:
            StepResult containing the output and execution metadata
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(step_id='{self.step_id}', step_type={self.step_type})"
