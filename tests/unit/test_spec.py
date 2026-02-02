"""Unit tests for pipeline specification models."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from rag_os.core.types import StepType
from rag_os.core.spec import StepSpec, PipelineSpec, RetryPolicy


class TestStepSpec:
    """Tests for StepSpec model."""

    def test_minimal_step_spec(self):
        """StepSpec can be created with minimal required fields."""
        spec = StepSpec(
            step_id="test_step",
            step_type=StepType.INGESTION,
            step_class="TestIngestionStep",
        )

        assert spec.step_id == "test_step"
        assert spec.step_type == StepType.INGESTION
        assert spec.step_class == "TestIngestionStep"
        assert spec.config == {}
        assert spec.dependencies == []
        assert spec.enabled is True

    def test_full_step_spec(self):
        """StepSpec can be created with all fields."""
        spec = StepSpec(
            step_id="embed_step",
            step_type=StepType.EMBEDDING,
            step_class="OpenAIEmbeddingStep",
            config={"model": "text-embedding-3-small"},
            dependencies=["chunk_step"],
            retry_policy={"max_retries": 3},
            fallback_step="fallback_embed",
            enabled=True,
        )

        assert spec.step_id == "embed_step"
        assert spec.config == {"model": "text-embedding-3-small"}
        assert spec.dependencies == ["chunk_step"]
        assert spec.retry_policy == {"max_retries": 3}
        assert spec.fallback_step == "fallback_embed"

    def test_step_id_validation_valid(self):
        """Valid step IDs are accepted."""
        valid_ids = ["step1", "my_step", "step-1", "Step_Name_123"]
        for step_id in valid_ids:
            spec = StepSpec(
                step_id=step_id,
                step_type=StepType.CHUNKING,
                step_class="TestStep",
            )
            assert spec.step_id == step_id

    def test_step_id_validation_invalid(self):
        """Invalid step IDs are rejected."""
        invalid_ids = ["step.1", "step/name", "step name", "step@1"]
        for step_id in invalid_ids:
            with pytest.raises(ValueError):
                StepSpec(
                    step_id=step_id,
                    step_type=StepType.CHUNKING,
                    step_class="TestStep",
                )

    def test_step_id_empty_rejected(self):
        """Empty step ID is rejected."""
        with pytest.raises(ValueError):
            StepSpec(
                step_id="",
                step_type=StepType.CHUNKING,
                step_class="TestStep",
            )

    def test_step_type_from_string(self):
        """Step type can be provided as string."""
        spec = StepSpec(
            step_id="test",
            step_type="chunking",  # type: ignore
            step_class="TestStep",
        )
        assert spec.step_type == StepType.CHUNKING


class TestRetryPolicy:
    """Tests for RetryPolicy model."""

    def test_default_values(self):
        """RetryPolicy has sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_seconds == 1.0
        assert policy.backoff_multiplier == 2.0
        assert policy.retry_on_errors == []

    def test_custom_values(self):
        """RetryPolicy accepts custom values."""
        policy = RetryPolicy(
            max_retries=5,
            backoff_seconds=0.5,
            backoff_multiplier=3.0,
            retry_on_errors=["TimeoutError", "ConnectionError"],
        )
        assert policy.max_retries == 5
        assert policy.backoff_seconds == 0.5
        assert policy.retry_on_errors == ["TimeoutError", "ConnectionError"]

    def test_max_retries_bounds(self):
        """max_retries must be between 0 and 10."""
        with pytest.raises(ValueError):
            RetryPolicy(max_retries=-1)
        with pytest.raises(ValueError):
            RetryPolicy(max_retries=11)


class TestPipelineSpec:
    """Tests for PipelineSpec model."""

    def test_minimal_pipeline_spec(self):
        """PipelineSpec can be created with minimal required fields."""
        spec = PipelineSpec(
            name="test-pipeline",
            version="1.0.0",
            steps=[
                StepSpec(
                    step_id="step1",
                    step_type=StepType.INGESTION,
                    step_class="TestStep",
                )
            ],
        )

        assert spec.name == "test-pipeline"
        assert spec.version == "1.0.0"
        assert len(spec.steps) == 1
        assert spec.description == ""
        assert spec.metadata == {}

    def test_full_pipeline_spec(self):
        """PipelineSpec can be created with all fields."""
        spec = PipelineSpec(
            name="full-pipeline",
            version="2.1.0",
            description="A complete pipeline",
            metadata={"author": "test"},
            default_config={"log_level": "debug"},
            steps=[
                StepSpec(step_id="s1", step_type=StepType.INGESTION, step_class="Step1"),
                StepSpec(step_id="s2", step_type=StepType.CHUNKING, step_class="Step2"),
            ],
        )

        assert spec.description == "A complete pipeline"
        assert spec.metadata == {"author": "test"}
        assert spec.default_config == {"log_level": "debug"}

    def test_version_format_valid(self):
        """Valid semantic versions are accepted."""
        valid_versions = ["1.0.0", "0.0.1", "10.20.30", "123.456.789"]
        for version in valid_versions:
            spec = PipelineSpec(
                name="test",
                version=version,
                steps=[StepSpec(step_id="s1", step_type=StepType.INGESTION, step_class="S")],
            )
            assert spec.version == version

    def test_version_format_invalid(self):
        """Invalid version formats are rejected."""
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "1.0.0.0"]
        for version in invalid_versions:
            with pytest.raises(ValueError):
                PipelineSpec(
                    name="test",
                    version=version,
                    steps=[StepSpec(step_id="s1", step_type=StepType.INGESTION, step_class="S")],
                )

    def test_empty_steps_rejected(self):
        """Pipeline with no steps is rejected."""
        with pytest.raises(ValueError):
            PipelineSpec(name="test", version="1.0.0", steps=[])

    def test_duplicate_step_ids_rejected(self):
        """Duplicate step IDs are rejected."""
        with pytest.raises(ValueError, match="Duplicate step IDs"):
            PipelineSpec(
                name="test",
                version="1.0.0",
                steps=[
                    StepSpec(step_id="step1", step_type=StepType.INGESTION, step_class="S1"),
                    StepSpec(step_id="step1", step_type=StepType.CHUNKING, step_class="S2"),
                ],
            )

    def test_invalid_dependency_rejected(self):
        """Dependencies to non-existent steps are rejected."""
        with pytest.raises(ValueError, match="non-existent step"):
            PipelineSpec(
                name="test",
                version="1.0.0",
                steps=[
                    StepSpec(
                        step_id="step1",
                        step_type=StepType.CHUNKING,
                        step_class="S1",
                        dependencies=["nonexistent"],
                    ),
                ],
            )

    def test_invalid_fallback_rejected(self):
        """Fallback to non-existent step is rejected."""
        with pytest.raises(ValueError, match="non-existent fallback"):
            PipelineSpec(
                name="test",
                version="1.0.0",
                steps=[
                    StepSpec(
                        step_id="step1",
                        step_type=StepType.LLM_EXECUTION,
                        step_class="S1",
                        fallback_step="nonexistent",
                    ),
                ],
            )

    def test_get_step(self):
        """get_step returns step by ID."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(step_id="s1", step_type=StepType.INGESTION, step_class="S1"),
                StepSpec(step_id="s2", step_type=StepType.CHUNKING, step_class="S2"),
            ],
        )

        assert spec.get_step("s1") is not None
        assert spec.get_step("s1").step_class == "S1"
        assert spec.get_step("s2").step_class == "S2"
        assert spec.get_step("nonexistent") is None

    def test_get_enabled_steps(self):
        """get_enabled_steps filters disabled steps."""
        spec = PipelineSpec(
            name="test",
            version="1.0.0",
            steps=[
                StepSpec(step_id="s1", step_type=StepType.INGESTION, step_class="S1", enabled=True),
                StepSpec(step_id="s2", step_type=StepType.CHUNKING, step_class="S2", enabled=False),
                StepSpec(step_id="s3", step_type=StepType.EMBEDDING, step_class="S3", enabled=True),
            ],
        )

        enabled = spec.get_enabled_steps()
        assert len(enabled) == 2
        assert enabled[0].step_id == "s1"
        assert enabled[1].step_id == "s3"


class TestPipelineSpecIO:
    """Tests for PipelineSpec file I/O."""

    def test_from_yaml(self, tmp_path: Path):
        """PipelineSpec can be loaded from YAML."""
        yaml_content = """
name: yaml-pipeline
version: "1.0.0"
steps:
  - step_id: step1
    step_type: ingestion
    step_class: TestStep
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        spec = PipelineSpec.from_yaml(yaml_file)
        assert spec.name == "yaml-pipeline"
        assert spec.version == "1.0.0"
        assert len(spec.steps) == 1

    def test_from_json(self, tmp_path: Path):
        """PipelineSpec can be loaded from JSON."""
        json_content = {
            "name": "json-pipeline",
            "version": "2.0.0",
            "steps": [
                {"step_id": "step1", "step_type": "chunking", "step_class": "TestStep"}
            ],
        }
        json_file = tmp_path / "pipeline.json"
        json_file.write_text(json.dumps(json_content))

        spec = PipelineSpec.from_json(json_file)
        assert spec.name == "json-pipeline"
        assert spec.version == "2.0.0"

    def test_from_dict(self):
        """PipelineSpec can be created from dict."""
        data = {
            "name": "dict-pipeline",
            "version": "1.0.0",
            "steps": [
                {"step_id": "s1", "step_type": "embedding", "step_class": "EmbedStep"}
            ],
        }

        spec = PipelineSpec.from_dict(data)
        assert spec.name == "dict-pipeline"

    def test_to_yaml(self, tmp_path: Path):
        """PipelineSpec can be saved to YAML."""
        spec = PipelineSpec(
            name="save-test",
            version="1.0.0",
            steps=[StepSpec(step_id="s1", step_type=StepType.RETRIEVAL, step_class="S1")],
        )

        yaml_file = tmp_path / "output.yaml"
        spec.to_yaml(yaml_file)

        # Verify file was created and can be read back
        loaded = PipelineSpec.from_yaml(yaml_file)
        assert loaded.name == "save-test"

    def test_to_json(self, tmp_path: Path):
        """PipelineSpec can be saved to JSON."""
        spec = PipelineSpec(
            name="save-test",
            version="1.0.0",
            steps=[StepSpec(step_id="s1", step_type=StepType.RERANKING, step_class="S1")],
        )

        json_file = tmp_path / "output.json"
        spec.to_json(json_file)

        # Verify file was created and can be read back
        loaded = PipelineSpec.from_json(json_file)
        assert loaded.name == "save-test"

    def test_to_dict(self):
        """PipelineSpec can be converted to dict."""
        spec = PipelineSpec(
            name="dict-test",
            version="1.0.0",
            description="Test description",
            steps=[StepSpec(step_id="s1", step_type=StepType.INGESTION, step_class="S1")],
        )

        data = spec.to_dict()
        assert data["name"] == "dict-test"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Test description"
        assert len(data["steps"]) == 1

    def test_from_yaml_file_not_found(self):
        """from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            PipelineSpec.from_yaml("/nonexistent/path/pipeline.yaml")

    def test_from_json_file_not_found(self):
        """from_json raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            PipelineSpec.from_json("/nonexistent/path/pipeline.json")

    def test_from_yaml_empty_file(self, tmp_path: Path):
        """from_yaml raises ValueError for empty file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(ValueError, match="Empty YAML"):
            PipelineSpec.from_yaml(yaml_file)

    def test_from_yaml_invalid_schema(self, tmp_path: Path):
        """from_yaml raises ValueError for invalid schema."""
        yaml_content = """
name: test
# Missing version and steps
"""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError):
            PipelineSpec.from_yaml(yaml_file)


class TestExamplePipelines:
    """Tests that verify example pipeline files are valid."""

    def test_simple_pipeline_yaml(self):
        """Example simple_pipeline.yaml is valid."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "simple_pipeline.yaml"
        if example_path.exists():
            spec = PipelineSpec.from_yaml(example_path)
            assert spec.name == "simple-rag-pipeline"
            assert len(spec.steps) > 0

    def test_simple_pipeline_json(self):
        """Example simple_pipeline.json is valid."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "simple_pipeline.json"
        if example_path.exists():
            spec = PipelineSpec.from_json(example_path)
            assert spec.name == "simple-rag-pipeline"
            assert len(spec.steps) > 0
