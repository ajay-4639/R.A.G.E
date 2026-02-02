"""Pipeline specification models for defining RAG pipelines."""

from typing import Any
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from rag_os.core.types import StepType, ConfigType, MetadataType


class StepSpec(BaseModel):
    """
    Specification for a single step in a pipeline.

    Attributes:
        step_id: Unique identifier for this step within the pipeline
        step_type: The category of this step (e.g., INGESTION, CHUNKING)
        step_class: Fully qualified class name or registered name of the step
        config: Step-specific configuration
        dependencies: List of step_ids this step depends on (for DAG support)
        retry_policy: Optional retry configuration
        fallback_step: Optional step_id to use if this step fails
        enabled: Whether this step is enabled (for toggling)
    """

    step_id: str = Field(..., min_length=1, description="Unique step identifier")
    step_type: StepType = Field(..., description="Type of the step")
    step_class: str = Field(..., min_length=1, description="Step class name or registered name")
    config: ConfigType = Field(default_factory=dict, description="Step configuration")
    dependencies: list[str] = Field(default_factory=list, description="Dependent step IDs")
    retry_policy: dict[str, Any] | None = Field(default=None, description="Retry configuration")
    fallback_step: str | None = Field(default=None, description="Fallback step ID on failure")
    enabled: bool = Field(default=True, description="Whether step is enabled")

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, v: str) -> str:
        """Ensure step_id contains only valid characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("step_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


class RetryPolicy(BaseModel):
    """Configuration for step retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    backoff_seconds: float = Field(default=1.0, ge=0, description="Initial backoff in seconds")
    backoff_multiplier: float = Field(default=2.0, ge=1.0, description="Backoff multiplier")
    retry_on_errors: list[str] = Field(default_factory=list, description="Error types to retry on")


class PipelineSpec(BaseModel):
    """
    Complete specification for a RAG pipeline.

    A pipeline spec defines the entire configuration of a RAG pipeline,
    including all steps, their order, and global settings.

    Attributes:
        name: Human-readable name for the pipeline
        version: Semantic version string (e.g., "1.0.0")
        description: Optional description of what this pipeline does
        steps: Ordered list of step specifications
        metadata: Additional metadata for the pipeline
        default_config: Default configuration applied to all steps
    """

    name: str = Field(..., min_length=1, description="Pipeline name")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    description: str = Field(default="", description="Pipeline description")
    steps: list[StepSpec] = Field(..., min_length=1, description="Pipeline steps")
    metadata: MetadataType = Field(default_factory=dict, description="Pipeline metadata")
    default_config: ConfigType = Field(default_factory=dict, description="Default step config")

    @field_validator("steps")
    @classmethod
    def validate_unique_step_ids(cls, v: list[StepSpec]) -> list[StepSpec]:
        """Ensure all step IDs are unique within the pipeline."""
        step_ids = [step.step_id for step in v]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]
            raise ValueError(f"Duplicate step IDs found: {set(duplicates)}")
        return v

    @field_validator("steps")
    @classmethod
    def validate_dependencies_exist(cls, v: list[StepSpec]) -> list[StepSpec]:
        """Ensure all dependencies reference existing steps."""
        step_ids = {step.step_id for step in v}
        for step in v:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.step_id}' depends on non-existent step '{dep}'"
                    )
        return v

    @field_validator("steps")
    @classmethod
    def validate_fallbacks_exist(cls, v: list[StepSpec]) -> list[StepSpec]:
        """Ensure all fallback steps reference existing steps."""
        step_ids = {step.step_id for step in v}
        for step in v:
            if step.fallback_step and step.fallback_step not in step_ids:
                raise ValueError(
                    f"Step '{step.step_id}' has non-existent fallback '{step.fallback_step}'"
                )
        return v

    def get_step(self, step_id: str) -> StepSpec | None:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_enabled_steps(self) -> list[StepSpec]:
        """Get only enabled steps in order."""
        return [step for step in self.steps if step.enabled]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineSpec":
        """
        Load a pipeline spec from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Parsed PipelineSpec instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the YAML is invalid or doesn't match schema
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline spec file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty YAML file: {path}")

        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineSpec":
        """
        Load a pipeline spec from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Parsed PipelineSpec instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON is invalid or doesn't match schema
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline spec file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineSpec":
        """
        Create a pipeline spec from a dictionary.

        Args:
            data: Dictionary containing pipeline specification

        Returns:
            Parsed PipelineSpec instance
        """
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save the pipeline spec to a YAML file.

        Args:
            path: Path to save the YAML file
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        """
        Save the pipeline spec to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        import json

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert the pipeline spec to a dictionary."""
        return self.model_dump(mode="json")
