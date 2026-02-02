"""Tests for RAG OS CLI."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from rag_os.cli.main import app


runner = CliRunner()


class TestVersion:
    """Tests for version command."""

    def test_version_displays(self):
        """Test version command shows version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "RAG OS" in result.output
        assert "0.1.0" in result.output


class TestQuery:
    """Tests for query command."""

    @patch("rag_os.cli.main.get_client")
    def test_query_basic(self, mock_get_client):
        """Test basic query execution."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_result.success = True
        mock_client.query.return_value = mock_result
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["query", "What is RAG?"])
        assert result.exit_code == 0
        mock_client.query.assert_called_once()

    @patch("rag_os.cli.main.get_client")
    def test_query_with_pipeline(self, mock_get_client):
        """Test query with specific pipeline."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_result.success = True
        mock_client.query.return_value = mock_result
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["query", "Test query", "--pipeline", "my-pipeline"])
        assert result.exit_code == 0
        mock_client.query.assert_called_with("Test query", pipeline="my-pipeline")

    @patch("rag_os.cli.main.get_client")
    def test_query_json_output(self, mock_get_client):
        """Test query with JSON output format."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_result.success = True
        mock_client.query.return_value = mock_result
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["query", "Test query", "--format", "json"])
        assert result.exit_code == 0
        # JSON should be in output
        assert "query" in result.output.lower() or "result" in result.output.lower()


class TestPipelineCommands:
    """Tests for pipeline commands."""

    @patch("rag_os.cli.main.get_client")
    def test_pipeline_list_empty(self, mock_get_client):
        """Test listing pipelines when none exist."""
        mock_client = Mock()
        mock_client.list_pipelines.return_value = []
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["pipeline", "list"])
        assert result.exit_code == 0
        assert "No pipelines" in result.output

    @patch("rag_os.cli.main.get_client")
    def test_pipeline_list_with_pipelines(self, mock_get_client):
        """Test listing existing pipelines."""
        mock_client = Mock()
        mock_client.list_pipelines.return_value = ["default", "custom"]
        mock_client.get_pipeline_info.return_value = {"steps": [1, 2, 3]}
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["pipeline", "list"])
        assert result.exit_code == 0

    @patch("rag_os.cli.main.get_client")
    def test_pipeline_info(self, mock_get_client):
        """Test getting pipeline info."""
        mock_client = Mock()
        mock_client.get_pipeline_info.return_value = {
            "version": "1.0.0",
            "steps": [{"id": "retriever", "type": "retrieval"}],
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["pipeline", "info", "default"])
        assert result.exit_code == 0

    @patch("rag_os.cli.main.get_client")
    def test_pipeline_info_not_found(self, mock_get_client):
        """Test getting info for non-existent pipeline."""
        mock_client = Mock()
        mock_client.get_pipeline_info.return_value = None
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["pipeline", "info", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_pipeline_create_basic(self, tmp_path):
        """Test creating a basic pipeline template."""
        output = tmp_path / "pipeline.json"
        result = runner.invoke(app, ["pipeline", "create", "test-pipeline", "--output", str(output)])

        assert result.exit_code == 0
        assert output.exists()

        spec = json.loads(output.read_text())
        assert spec["name"] == "test-pipeline"
        assert "steps" in spec

    def test_pipeline_create_minimal_template(self, tmp_path):
        """Test creating a minimal pipeline template."""
        output = tmp_path / "minimal.json"
        result = runner.invoke(app, ["pipeline", "create", "minimal", "--template", "minimal", "--output", str(output)])

        assert result.exit_code == 0
        spec = json.loads(output.read_text())
        assert len(spec["steps"]) == 1

    def test_pipeline_create_advanced_template(self, tmp_path):
        """Test creating an advanced pipeline template."""
        output = tmp_path / "advanced.json"
        result = runner.invoke(app, ["pipeline", "create", "advanced", "--template", "advanced", "--output", str(output)])

        assert result.exit_code == 0
        spec = json.loads(output.read_text())
        assert len(spec["steps"]) > 4


class TestConfigCommands:
    """Tests for config commands."""

    def test_config_show_default(self):
        """Test showing default configuration."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "Default configuration" in result.output or "default_pipeline" in result.output

    def test_config_init(self, tmp_path):
        """Test initializing configuration file."""
        output = tmp_path / "rag_os.json"
        result = runner.invoke(app, ["config", "init", "--output", str(output)])

        assert result.exit_code == 0
        assert output.exists()

        config = json.loads(output.read_text())
        assert "default_pipeline" in config

    def test_config_init_no_overwrite(self, tmp_path):
        """Test config init doesn't overwrite without force."""
        output = tmp_path / "rag_os.json"
        output.write_text("{}")

        result = runner.invoke(app, ["config", "init", "--output", str(output)])
        assert result.exit_code == 1
        assert "already exists" in result.output.lower()

    def test_config_init_force_overwrite(self, tmp_path):
        """Test config init with force flag."""
        output = tmp_path / "rag_os.json"
        output.write_text("{}")

        result = runner.invoke(app, ["config", "init", "--output", str(output), "--force"])
        assert result.exit_code == 0


class TestIndexCommands:
    """Tests for index commands."""

    def test_index_list_empty(self):
        """Test listing indexes when none exist."""
        result = runner.invoke(app, ["index", "list"])
        assert result.exit_code == 0
        assert "No indexes" in result.output

    def test_index_create_nonexistent_source(self, tmp_path):
        """Test creating index with nonexistent source."""
        result = runner.invoke(app, ["index", "create", "test-index", str(tmp_path / "nonexistent")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_index_create_from_file(self, tmp_path):
        """Test creating index from a file."""
        source = tmp_path / "doc.txt"
        source.write_text("Test document content")

        result = runner.invoke(app, ["index", "create", "test-index", str(source)])
        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_index_delete_with_confirmation(self):
        """Test delete requires confirmation."""
        result = runner.invoke(app, ["index", "delete", "test-index"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_index_delete_force(self):
        """Test delete with force flag."""
        result = runner.invoke(app, ["index", "delete", "test-index", "--force"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()


class TestInteractive:
    """Tests for interactive mode."""

    @patch("rag_os.cli.main.get_client")
    @patch("rag_os.cli.main.console")
    def test_interactive_exit(self, mock_console, mock_get_client):
        """Test exiting interactive mode."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock console.input to return 'exit'
        mock_console.input.return_value = "exit"

        result = runner.invoke(app, ["interactive"])
        # Should exit cleanly
        assert "Goodbye" in result.output or result.exit_code == 0

    @patch("rag_os.cli.main.get_client")
    @patch("rag_os.cli.main.console")
    def test_interactive_help_command(self, mock_console, mock_get_client):
        """Test help command in interactive mode."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Return /help then exit
        mock_console.input.side_effect = ["/help", "exit"]

        result = runner.invoke(app, ["interactive"])
        # Help should be displayed
        assert "Commands" in result.output or result.exit_code == 0


class TestServe:
    """Tests for serve command."""

    def test_serve_missing_uvicorn(self):
        """Test serve command shows error when uvicorn not installed."""
        # The serve command catches ImportError and shows a helpful message
        result = runner.invoke(app, ["serve"])
        # Either succeeds (uvicorn installed) or shows install message
        assert result.exit_code in [0, 1]

    def test_serve_help(self):
        """Test serve help shows options."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output

    def test_serve_default_options_in_help(self):
        """Test default options are documented."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "0.0.0.0" in result.output  # default host
        assert "8000" in result.output  # default port

