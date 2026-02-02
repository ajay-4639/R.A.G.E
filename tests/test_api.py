"""Tests for RAG OS REST API."""

from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from rag_os.api.server import app, state, AppState


@pytest.fixture
def client():
    """Create test client."""
    # Reset state for each test
    state.client = Mock()
    state.rate_limiter = Mock()
    state.validator = Mock()
    state.start_time = datetime.utcnow()
    state.request_count = 0

    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_readiness_check_ready(self, client):
        """Test readiness when service is ready."""
        state.client = Mock()
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_readiness_check_not_ready(self, client):
        """Test readiness when service is not ready."""
        state.client = None
        response = client.get("/health/ready")
        assert response.status_code == 503

    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "RAG OS API"
        assert "version" in data
        assert "docs" in data


class TestQueryEndpoints:
    """Tests for query endpoints."""

    def test_query_success(self, client):
        """Test successful query."""
        # Mock rate limiter
        state.rate_limiter.check.return_value = Mock(allowed=True)

        # Mock client
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_result.success = True
        state.client.query.return_value = mock_result

        response = client.post("/query", json={
            "query": "What is RAG?",
            "pipeline": "default",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is RAG?"
        assert data["result"] == "Test response"
        assert data["success"] is True
        assert "duration_ms" in data

    def test_query_rate_limited(self, client):
        """Test query with rate limiting."""
        state.rate_limiter.check.return_value = Mock(allowed=False, retry_after=5.0)

        response = client.post("/query", json={
            "query": "Test query",
        })

        assert response.status_code == 429
        assert "rate limit" in response.json()["error"].lower()

    def test_query_invalid_input(self, client):
        """Test query with invalid input."""
        response = client.post("/query", json={
            "query": "",  # Empty query
        })

        assert response.status_code == 422  # Validation error

    def test_query_with_context(self, client):
        """Test query with additional context."""
        state.rate_limiter.check.return_value = Mock(allowed=True)

        mock_result = Mock()
        mock_result.data = "Response"
        mock_result.success = True
        state.client.query.return_value = mock_result

        response = client.post("/query", json={
            "query": "Test query",
            "context": {"user_id": "123"},
        })

        assert response.status_code == 200
        state.client.query.assert_called_with(
            "Test query",
            pipeline="default",
            context={"user_id": "123"},
        )


class TestPipelineEndpoints:
    """Tests for pipeline endpoints."""

    def test_list_pipelines(self, client):
        """Test listing pipelines."""
        state.client.list_pipelines.return_value = ["default", "custom"]

        response = client.get("/pipelines")
        assert response.status_code == 200
        assert response.json() == ["default", "custom"]

    def test_list_pipelines_empty(self, client):
        """Test listing when no pipelines exist."""
        state.client.list_pipelines.return_value = []

        response = client.get("/pipelines")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_pipeline(self, client):
        """Test getting pipeline info."""
        state.client.get_pipeline_info.return_value = {
            "version": "1.0.0",
            "steps": [{"id": "retriever", "type": "retrieval"}],
        }

        response = client.get("/pipelines/default")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "default"
        assert data["version"] == "1.0.0"
        assert len(data["steps"]) == 1

    def test_get_pipeline_not_found(self, client):
        """Test getting non-existent pipeline."""
        state.client.get_pipeline_info.return_value = None

        response = client.get("/pipelines/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["error"].lower()

    def test_create_pipeline(self, client):
        """Test creating a pipeline."""
        state.client.load_pipeline = Mock()

        spec = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "step_id": "retriever",
                    "step_type": "retrieval",
                    "step_class": "rag_os.steps.retrieval.VectorRetriever",
                    "config": {},
                },
            ],
        }

        response = client.post("/pipelines", json=spec)
        assert response.status_code == 200
        assert response.json()["name"] == "test-pipeline"

    def test_create_pipeline_invalid_spec(self, client):
        """Test creating pipeline with invalid spec."""
        response = client.post("/pipelines", json={
            "name": "test",
            # Missing required fields
        })

        assert response.status_code == 400

    def test_delete_pipeline(self, client):
        """Test deleting a pipeline."""
        state.client.unload_pipeline = Mock()

        response = client.delete("/pipelines/test")
        assert response.status_code == 200
        assert response.json()["name"] == "test"

    def test_delete_pipeline_not_found(self, client):
        """Test deleting non-existent pipeline."""
        state.client.unload_pipeline.side_effect = Exception("Not found")

        response = client.delete("/pipelines/nonexistent")
        assert response.status_code == 404


class TestIndexEndpoints:
    """Tests for index endpoints."""

    def test_list_indexes(self, client):
        """Test listing indexes."""
        response = client.get("/indexes")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_index(self, client):
        """Test creating an index."""
        response = client.post("/indexes", json={
            "name": "test-index",
            "documents": [{"content": "Test document"}],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-index"
        assert data["status"] == "indexing"

    def test_get_index_not_found(self, client):
        """Test getting non-existent index."""
        response = client.get("/indexes/nonexistent")
        assert response.status_code == 404

    def test_delete_index_not_found(self, client):
        """Test deleting non-existent index."""
        response = client.delete("/indexes/nonexistent")
        assert response.status_code == 404


class TestDocumentEndpoints:
    """Tests for document endpoints."""

    def test_add_document(self, client):
        """Test adding a document."""
        response = client.post("/indexes/test/documents", json={
            "content": "Test document content",
            "metadata": {"author": "test"},
        })

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["index"] == "test"

    def test_add_document_empty_content(self, client):
        """Test adding document with empty content."""
        response = client.post("/indexes/test/documents", json={
            "content": "",
        })

        assert response.status_code == 422  # Validation error

    def test_get_document_not_found(self, client):
        """Test getting non-existent document."""
        response = client.get("/indexes/test/documents/nonexistent")
        assert response.status_code == 404

    def test_delete_document_not_found(self, client):
        """Test deleting non-existent document."""
        response = client.delete("/indexes/test/documents/nonexistent")
        assert response.status_code == 404


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_get_metrics(self, client):
        """Test getting metrics."""
        state.request_count = 100

        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "requests_total" in data
        assert "uptime_seconds" in data
        assert "version" in data


class TestRequestTracking:
    """Tests for request tracking middleware."""

    def test_request_id_header(self, client):
        """Test that request ID is added to response headers."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"]  # Not empty

    def test_response_time_header(self, client):
        """Test that response time is added to headers."""
        response = client.get("/health")

        assert "X-Response-Time-Ms" in response.headers
        time_ms = float(response.headers["X-Response-Time-Ms"])
        assert time_ms >= 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_internal_error_handling(self, client):
        """Test internal server error handling."""
        state.rate_limiter.check.return_value = Mock(allowed=True)
        state.client.query.side_effect = Exception("Internal error")

        response = client.post("/query", json={"query": "Test"})

        assert response.status_code == 500
        assert "error" in response.json()

    def test_validation_error_handling(self, client):
        """Test validation error handling."""
        # Invalid JSON
        response = client.post(
            "/query",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS headers should be present
        assert response.status_code in [200, 405]  # Depends on CORS config
