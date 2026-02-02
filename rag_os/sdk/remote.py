"""Remote client for RAG OS - Connect to a hosted RAG OS server.

This module provides a thin HTTP client that communicates with a
RAG OS server running as a Docker container or hosted service.

Usage:
    # Connect to a running RAG OS server
    from rag_os import RemoteClient

    client = RemoteClient("http://localhost:8000")

    # Create a pipeline
    client.create_pipeline({
        "name": "my-rag",
        "version": "1.0.0",
        "steps": [...]
    })

    # Query it
    result = client.query("What is the refund policy?", pipeline="my-rag")
    print(result.answer)
    print(result.sources)

    # Or use the pipeline builder
    from rag_os import PipelineBuilder, StepBuilder
    from rag_os.core.types import StepType

    spec = (PipelineBuilder("my-pipeline")
        .add_ingestion_step("ingest", file_path="docs/")
        .add_chunking_step("chunk", depends_on="ingest", chunk_size=512)
        .add_llm_step("llm", depends_on="chunk", model="gpt-4o-mini")
        .build())

    client.deploy_pipeline(spec)
    result = client.query("Summarize the docs", pipeline="my-pipeline")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


@dataclass
class QueryResult:
    """Result from a RAG query.

    Attributes:
        answer: The generated answer text
        query: Original query string
        pipeline: Pipeline that was used
        success: Whether the query succeeded
        duration_ms: Processing time in milliseconds
        sources: Retrieved source chunks (if available)
        metadata: Additional result metadata
        request_id: Server-assigned request ID
    """
    answer: str
    query: str
    pipeline: str
    success: bool
    duration_ms: float
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"QueryResult({status}, {self.duration_ms:.0f}ms, answer={self.answer[:80]!r}...)"


@dataclass
class PipelineInfo:
    """Information about a deployed pipeline.

    Attributes:
        name: Pipeline name
        version: Pipeline version
        description: Pipeline description
        steps: List of step info dicts
        status: Pipeline status (active, inactive)
    """
    name: str
    version: str
    description: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: str = "active"


@dataclass
class ServerHealth:
    """Health status of the RAG OS server.

    Attributes:
        status: Health status string
        version: Server version
        uptime_seconds: Server uptime in seconds
    """
    status: str
    version: str
    uptime_seconds: float


class RemoteClientError(Exception):
    """Error from the RAG OS remote client."""

    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class RemoteClient:
    """HTTP client for a remote RAG OS server.

    Connects to a RAG OS instance running as a Docker container
    or hosted service and provides the same interface as the local
    RAGClient.

    Usage:
        # Connect to server
        client = RemoteClient("http://localhost:8000")

        # Check server health
        health = client.health()
        print(health.status)  # "healthy"

        # Deploy a pipeline
        client.create_pipeline({
            "name": "qa-pipeline",
            "version": "1.0.0",
            "steps": [...]
        })

        # Query the pipeline
        result = client.query("What is RAG?", pipeline="qa-pipeline")
        print(result.answer)

        # List pipelines
        pipelines = client.list_pipelines()
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ):
        """Initialize remote client.

        Args:
            server_url: URL of the RAG OS server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            headers: Additional HTTP headers

        Raises:
            ImportError: If httpx is not installed
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for RemoteClient. "
                "Install it with: pip install rag-os[remote] or pip install httpx"
            )

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        # Build default headers
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": "rag-os-client/0.1.0",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self._headers.update(headers)

        self._client = httpx.Client(
            base_url=self.server_url,
            headers=self._headers,
            timeout=timeout,
        )

    def _request(
        self,
        method: str,
        path: str,
        json_data: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request to the server.

        Args:
            method: HTTP method
            path: API path
            json_data: JSON body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            RemoteClientError: On request failure
        """
        try:
            response = self._client.request(
                method=method,
                url=path,
                json=json_data,
                params=params,
            )

            if response.status_code >= 400:
                try:
                    error_body = response.json()
                    detail = error_body.get("detail") or error_body.get("error", "")
                except Exception:
                    detail = response.text

                raise RemoteClientError(
                    f"Request failed: {method} {path} -> {response.status_code}",
                    status_code=response.status_code,
                    detail=detail,
                )

            if response.status_code == 204:
                return None

            return response.json()

        except httpx.ConnectError:
            raise RemoteClientError(
                f"Cannot connect to RAG OS server at {self.server_url}. "
                "Is the server running?"
            )
        except httpx.TimeoutException:
            raise RemoteClientError(
                f"Request timed out after {self.timeout}s. "
                "Try increasing the timeout or check server load."
            )

    # ---- Health ----

    def health(self) -> ServerHealth:
        """Check server health.

        Returns:
            ServerHealth with status, version, and uptime
        """
        data = self._request("GET", "/health")
        return ServerHealth(
            status=data["status"],
            version=data["version"],
            uptime_seconds=data.get("uptime_seconds", 0),
        )

    def is_healthy(self) -> bool:
        """Quick health check.

        Returns:
            True if server is healthy
        """
        try:
            h = self.health()
            return h.status == "healthy"
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 60.0, interval: float = 2.0) -> bool:
        """Wait until the server is ready.

        Args:
            timeout: Maximum time to wait in seconds
            interval: Time between checks in seconds

        Returns:
            True if server became ready, False if timed out
        """
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self.is_healthy():
                return True
            time.sleep(interval)
        return False

    # ---- Pipelines ----

    def create_pipeline(self, spec: dict[str, Any] | Any) -> str:
        """Create/deploy a pipeline on the server.

        Args:
            spec: Pipeline specification dict or PipelineSpec object

        Returns:
            Pipeline name
        """
        if hasattr(spec, "to_dict"):
            spec_dict = spec.to_dict()
        elif hasattr(spec, "model_dump"):
            spec_dict = spec.model_dump(mode="json")
        else:
            spec_dict = spec

        data = self._request("POST", "/pipelines", json_data=spec_dict)
        return data.get("name", spec_dict.get("name", ""))

    def deploy_pipeline(self, spec: Any) -> str:
        """Alias for create_pipeline. Deploy a pipeline spec to the server.

        Args:
            spec: PipelineSpec object or dict

        Returns:
            Pipeline name
        """
        return self.create_pipeline(spec)

    def list_pipelines(self) -> list[str]:
        """List all deployed pipelines.

        Returns:
            List of pipeline names
        """
        return self._request("GET", "/pipelines")

    def get_pipeline(self, name: str) -> PipelineInfo:
        """Get information about a pipeline.

        Args:
            name: Pipeline name

        Returns:
            PipelineInfo object
        """
        data = self._request("GET", f"/pipelines/{name}")
        return PipelineInfo(
            name=data["name"],
            version=data.get("version", ""),
            description=data.get("description", ""),
            steps=data.get("steps", []),
            status=data.get("status", "active"),
        )

    def delete_pipeline(self, name: str) -> None:
        """Delete a pipeline from the server.

        Args:
            name: Pipeline name
        """
        self._request("DELETE", f"/pipelines/{name}")

    # ---- Querying ----

    def query(
        self,
        query: str,
        pipeline: str = "default",
        context: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Query a RAG pipeline.

        Args:
            query: The question/query text
            pipeline: Pipeline name to use
            context: Additional context data

        Returns:
            QueryResult with answer, sources, and metadata
        """
        body: dict[str, Any] = {
            "query": query,
            "pipeline": pipeline,
        }
        if context:
            body["context"] = context

        data = self._request("POST", "/query", json_data=body)

        # Extract result data
        result = data.get("result", {})
        if isinstance(result, dict):
            answer = result.get("answer", result.get("content", str(result)))
            sources = result.get("sources", [])
            metadata = result.get("metadata", {})
        elif isinstance(result, str):
            answer = result
            sources = []
            metadata = {}
        else:
            answer = str(result)
            sources = []
            metadata = {}

        return QueryResult(
            answer=answer,
            query=query,
            pipeline=pipeline,
            success=data.get("success", True),
            duration_ms=data.get("duration_ms", 0),
            sources=sources,
            metadata=metadata,
            request_id=data.get("id", ""),
        )

    # ---- Documents ----

    def add_documents(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add documents to an index for ingestion.

        Args:
            index_name: Name of the index
            documents: List of document dicts with 'content' and optional 'metadata'
            config: Optional indexing configuration

        Returns:
            Indexing result with status and counts
        """
        body: dict[str, Any] = {
            "name": index_name,
            "documents": documents,
        }
        if config:
            body["config"] = config

        return self._request("POST", "/indexes", json_data=body)

    # ---- Metrics ----

    def metrics(self) -> dict[str, Any]:
        """Get server metrics.

        Returns:
            Dict of metric values
        """
        return self._request("GET", "/metrics")

    # ---- Pipeline from file ----

    def deploy_from_file(self, path: str | Path) -> str:
        """Deploy a pipeline from a local JSON or YAML file.

        Args:
            path: Path to pipeline spec file

        Returns:
            Pipeline name
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        if path.suffix in (".yaml", ".yml"):
            import yaml
            with open(path, "r") as f:
                spec_dict = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                spec_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return self.create_pipeline(spec_dict)

    # ---- Context manager ----

    def __enter__(self) -> "RemoteClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client connection."""
        self._client.close()

    def __repr__(self) -> str:
        return f"RemoteClient({self.server_url!r})"
