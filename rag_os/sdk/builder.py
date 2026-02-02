"""Pipeline builder for RAG OS - Fluent API for building pipelines."""

from typing import Any
from dataclasses import dataclass, field

from rag_os.core.spec import PipelineSpec, StepSpec
from rag_os.core.types import StepType


@dataclass
class StepBuilder:
    """Builder for constructing pipeline steps.

    Provides a fluent API for step configuration.

    Usage:
        step = (StepBuilder("embed_1")
            .with_class("OpenAIEmbeddingStep")
            .with_config(model="text-embedding-3-small")
            .depends_on("chunk_1")
            .build())
    """
    step_id: str
    _step_class: str = ""
    _step_type: StepType | None = None
    _config: dict[str, Any] = field(default_factory=dict)
    _dependencies: list[str] = field(default_factory=list)
    _enabled: bool = True
    _fallback: str | None = None
    _retry_count: int = 0
    _timeout_ms: int | None = None

    def with_class(self, step_class: str) -> "StepBuilder":
        """Set the step class.

        Args:
            step_class: Name of the step class

        Returns:
            Self for chaining
        """
        self._step_class = step_class
        return self

    def with_type(self, step_type: StepType) -> "StepBuilder":
        """Set the step type.

        Args:
            step_type: Type of the step

        Returns:
            Self for chaining
        """
        self._step_type = step_type
        return self

    def with_config(self, **config: Any) -> "StepBuilder":
        """Set step configuration.

        Args:
            **config: Configuration key-value pairs

        Returns:
            Self for chaining
        """
        self._config.update(config)
        return self

    def depends_on(self, *step_ids: str) -> "StepBuilder":
        """Add step dependencies.

        Args:
            *step_ids: IDs of steps this step depends on

        Returns:
            Self for chaining
        """
        self._dependencies.extend(step_ids)
        return self

    def enabled(self, value: bool = True) -> "StepBuilder":
        """Set whether step is enabled.

        Args:
            value: Enabled state

        Returns:
            Self for chaining
        """
        self._enabled = value
        return self

    def disabled(self) -> "StepBuilder":
        """Disable the step.

        Returns:
            Self for chaining
        """
        return self.enabled(False)

    def with_fallback(self, fallback_id: str) -> "StepBuilder":
        """Set fallback step.

        Args:
            fallback_id: ID of fallback step

        Returns:
            Self for chaining
        """
        self._fallback = fallback_id
        return self

    def with_retry(self, count: int) -> "StepBuilder":
        """Set retry count.

        Args:
            count: Number of retries

        Returns:
            Self for chaining
        """
        self._retry_count = count
        return self

    def with_timeout(self, timeout_ms: int) -> "StepBuilder":
        """Set operation timeout.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Self for chaining
        """
        self._timeout_ms = timeout_ms
        return self

    def build(self) -> StepSpec:
        """Build the step specification.

        Returns:
            StepSpec instance
        """
        if not self._step_class:
            raise ValueError("Step class is required")

        # Default step type if not specified
        step_type = self._step_type or StepType.LLM_EXECUTION

        # Build retry policy if retry_count is set
        retry_policy = None
        if self._retry_count > 0:
            retry_policy = {"max_retries": self._retry_count}
        if self._timeout_ms is not None:
            if retry_policy is None:
                retry_policy = {}
            retry_policy["timeout_ms"] = self._timeout_ms

        return StepSpec(
            step_id=self.step_id,
            step_type=step_type,
            step_class=self._step_class,
            config=self._config if self._config else {},
            dependencies=self._dependencies if self._dependencies else [],
            enabled=self._enabled,
            fallback_step=self._fallback,
            retry_policy=retry_policy,
        )


class PipelineBuilder:
    """Builder for constructing pipelines.

    Provides a fluent API for pipeline configuration.

    Usage:
        pipeline = (PipelineBuilder("qa_pipeline")
            .with_description("Question answering pipeline")
            .add_step(
                StepBuilder("retrieve_1")
                .with_class("VectorRetrievalStep")
                .with_config(top_k=5)
                .build()
            )
            .add_step(
                StepBuilder("llm_1")
                .with_class("OpenAILLMStep")
                .depends_on("retrieve_1")
                .build()
            )
            .build())
    """

    def __init__(self, name: str):
        """Initialize pipeline builder.

        Args:
            name: Pipeline name
        """
        self._name = name
        self._version = "1.0.0"
        self._description = ""
        self._steps: list[StepSpec] = []
        self._metadata: dict[str, Any] = {}
        self._global_config: dict[str, Any] = {}

    def with_version(self, version: str) -> "PipelineBuilder":
        """Set pipeline version.

        Args:
            version: Version string

        Returns:
            Self for chaining
        """
        self._version = version
        return self

    def with_description(self, description: str) -> "PipelineBuilder":
        """Set pipeline description.

        Args:
            description: Description text

        Returns:
            Self for chaining
        """
        self._description = description
        return self

    def with_metadata(self, **metadata: Any) -> "PipelineBuilder":
        """Add metadata.

        Args:
            **metadata: Metadata key-value pairs

        Returns:
            Self for chaining
        """
        self._metadata.update(metadata)
        return self

    def with_global_config(self, **config: Any) -> "PipelineBuilder":
        """Set global configuration.

        Args:
            **config: Configuration key-value pairs

        Returns:
            Self for chaining
        """
        self._global_config.update(config)
        return self

    def add_step(self, step: StepSpec | StepBuilder) -> "PipelineBuilder":
        """Add a step to the pipeline.

        Args:
            step: Step specification or builder

        Returns:
            Self for chaining
        """
        if isinstance(step, StepBuilder):
            step = step.build()
        self._steps.append(step)
        return self

    def add_steps(self, *steps: StepSpec | StepBuilder) -> "PipelineBuilder":
        """Add multiple steps.

        Args:
            *steps: Step specifications or builders

        Returns:
            Self for chaining
        """
        for step in steps:
            self.add_step(step)
        return self

    def add_ingestion_step(
        self,
        step_id: str,
        step_class: str = "FileIngestionStep",
        **config: Any,
    ) -> "PipelineBuilder":
        """Add an ingestion step.

        Args:
            step_id: Step ID
            step_class: Ingestion step class name
            **config: Step configuration

        Returns:
            Self for chaining
        """
        return self.add_step(
            StepBuilder(step_id)
            .with_class(step_class)
            .with_type(StepType.INGESTION)
            .with_config(**config)
        )

    def add_chunking_step(
        self,
        step_id: str,
        step_class: str = "FixedSizeChunkingStep",
        depends_on: str | None = None,
        **config: Any,
    ) -> "PipelineBuilder":
        """Add a chunking step.

        Args:
            step_id: Step ID
            step_class: Chunking step class name
            depends_on: Dependency step ID
            **config: Step configuration

        Returns:
            Self for chaining
        """
        builder = (
            StepBuilder(step_id)
            .with_class(step_class)
            .with_type(StepType.CHUNKING)
            .with_config(**config)
        )
        if depends_on:
            builder.depends_on(depends_on)
        return self.add_step(builder)

    def add_embedding_step(
        self,
        step_id: str,
        step_class: str = "OpenAIEmbeddingStep",
        depends_on: str | None = None,
        **config: Any,
    ) -> "PipelineBuilder":
        """Add an embedding step.

        Args:
            step_id: Step ID
            step_class: Embedding step class name
            depends_on: Dependency step ID
            **config: Step configuration

        Returns:
            Self for chaining
        """
        builder = (
            StepBuilder(step_id)
            .with_class(step_class)
            .with_type(StepType.EMBEDDING)
            .with_config(**config)
        )
        if depends_on:
            builder.depends_on(depends_on)
        return self.add_step(builder)

    def add_retrieval_step(
        self,
        step_id: str,
        step_class: str = "VectorRetrievalStep",
        depends_on: str | None = None,
        **config: Any,
    ) -> "PipelineBuilder":
        """Add a retrieval step.

        Args:
            step_id: Step ID
            step_class: Retrieval step class name
            depends_on: Dependency step ID
            **config: Step configuration

        Returns:
            Self for chaining
        """
        builder = (
            StepBuilder(step_id)
            .with_class(step_class)
            .with_type(StepType.RETRIEVAL)
            .with_config(**config)
        )
        if depends_on:
            builder.depends_on(depends_on)
        return self.add_step(builder)

    def add_llm_step(
        self,
        step_id: str,
        step_class: str = "OpenAILLMStep",
        depends_on: str | None = None,
        **config: Any,
    ) -> "PipelineBuilder":
        """Add an LLM step.

        Args:
            step_id: Step ID
            step_class: LLM step class name
            depends_on: Dependency step ID
            **config: Step configuration

        Returns:
            Self for chaining
        """
        builder = (
            StepBuilder(step_id)
            .with_class(step_class)
            .with_type(StepType.LLM_EXECUTION)
            .with_config(**config)
        )
        if depends_on:
            builder.depends_on(depends_on)
        return self.add_step(builder)

    def build(self) -> PipelineSpec:
        """Build the pipeline specification.

        Returns:
            PipelineSpec instance
        """
        if not self._steps:
            raise ValueError("Pipeline must have at least one step")

        return PipelineSpec(
            name=self._name,
            version=self._version,
            description=self._description,
            steps=self._steps,
            metadata=self._metadata if self._metadata else {},
            default_config=self._global_config if self._global_config else {},
        )

    def to_yaml(self, path: str) -> None:
        """Save pipeline to YAML file.

        Args:
            path: File path
        """
        spec = self.build()
        spec.to_yaml(path)

    def to_json(self, path: str) -> None:
        """Save pipeline to JSON file.

        Args:
            path: File path
        """
        spec = self.build()
        spec.to_json(path)
