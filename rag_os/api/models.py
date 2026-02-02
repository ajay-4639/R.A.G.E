"""API request/response models for RAG OS."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=10000, description="The query text")
    pipeline: str = Field(default="default", description="Pipeline to use")
    context: Optional[dict[str, Any]] = Field(default=None, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversational queries")


class QueryResponse(BaseModel):
    """Query response model."""
    id: str = Field(..., description="Request ID")
    query: str = Field(..., description="Original query")
    result: Any = Field(..., description="Query result")
    pipeline: str = Field(..., description="Pipeline used")
    success: bool = Field(..., description="Whether query succeeded")
    duration_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")
    sources: Optional[list[dict[str, Any]]] = Field(default=None, description="Retrieved sources")


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    id: str
    chunk: str
    done: bool = False
    metadata: Optional[dict[str, Any]] = None


class PipelineSpec(BaseModel):
    """Pipeline specification for creation."""
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(default="1.0.0")
    steps: list[dict[str, Any]] = Field(..., min_items=1)
    config: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class PipelineInfo(BaseModel):
    """Pipeline information response."""
    name: str
    version: str
    steps: list[dict[str, Any]]
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class IndexSpec(BaseModel):
    """Index specification for creation."""
    name: str = Field(..., min_length=1, max_length=100)
    config: Optional[dict[str, Any]] = None
    embedding_model: str = Field(default="text-embedding-3-small")
    chunk_size: int = Field(default=512, ge=100, le=8000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class IndexInfo(BaseModel):
    """Index information response."""
    name: str
    document_count: int
    chunk_count: int
    embedding_model: str
    status: str
    created_at: str
    updated_at: str
    size_bytes: int


class DocumentSpec(BaseModel):
    """Document specification for ingestion."""
    content: str = Field(..., min_length=1)
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None
    doc_type: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information response."""
    id: str
    index: str
    source: Optional[str]
    chunk_count: int
    created_at: str
    metadata: Optional[dict[str, Any]] = None


class BatchIngestRequest(BaseModel):
    """Batch document ingestion request."""
    documents: list[DocumentSpec] = Field(..., min_items=1, max_items=100)
    async_processing: bool = Field(default=True)


class BatchIngestResponse(BaseModel):
    """Batch ingestion response."""
    job_id: str
    status: str
    total_documents: int
    processed: int = 0
    failed: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    uptime_seconds: float
    components: Optional[dict[str, str]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    code: Optional[str] = None


class MetricsResponse(BaseModel):
    """Metrics response model."""
    requests_total: int
    requests_per_second: float
    average_latency_ms: float
    error_rate: float
    uptime_seconds: float
    active_sessions: int
    cache_hit_rate: Optional[float] = None


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    created_at: str
    last_active: str
    query_count: int
    pipeline: str


class SearchRequest(BaseModel):
    """Semantic search request."""
    query: str = Field(..., min_length=1, max_length=10000)
    index: str = Field(..., description="Index to search")
    top_k: int = Field(default=10, ge=1, le=100)
    filter: Optional[dict[str, Any]] = Field(default=None, description="Metadata filter")
    min_score: Optional[float] = Field(default=None, ge=0, le=1)


class SearchResult(BaseModel):
    """Search result item."""
    id: str
    content: str
    score: float
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: list[SearchResult]
    total: int
    duration_ms: float
