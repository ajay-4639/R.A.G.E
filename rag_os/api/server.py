"""FastAPI server for RAG OS."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import rag_os
from rag_os.sdk.client import RAGClient, RAGClientConfig
from rag_os.security.rate_limit import RateLimiter, RateLimitConfig
from rag_os.security.validation import InputValidator, validate_query


# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=10000, description="The query text")
    pipeline: str = Field(default="default", description="Pipeline to use")
    context: Optional[dict[str, Any]] = Field(default=None, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Query response model."""
    id: str = Field(..., description="Request ID")
    query: str = Field(..., description="Original query")
    result: Any = Field(..., description="Query result")
    pipeline: str = Field(..., description="Pipeline used")
    success: bool = Field(..., description="Whether query succeeded")
    duration_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")


class PipelineInfo(BaseModel):
    """Pipeline information model."""
    name: str
    version: str
    description: str = ""
    steps: list[dict[str, Any]]
    status: str = "active"


class IndexRequest(BaseModel):
    """Index creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    documents: list[dict[str, Any]] = Field(..., min_length=1)
    config: Optional[dict[str, Any]] = None


class DocumentRequest(BaseModel):
    """Document ingestion request."""
    content: str = Field(..., min_length=1)
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# Global state
class AppState:
    """Application state container."""
    def __init__(self):
        self.client: Optional[RAGClient] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.validator: Optional[InputValidator] = None
        self.start_time: datetime = datetime.now()
        self.request_count: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    state.client = RAGClient()
    state.rate_limiter = RateLimiter(RateLimitConfig(
        requests_per_second=10.0,
        burst_size=20,
    ))
    state.validator = InputValidator()
    state.start_time = datetime.now()
    yield
    # Shutdown
    state.client = None


def create_app(config: Optional[dict[str, Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG OS API",
        description="REST API for RAG OS - A fully customizable RAG Operating System",
        version=rag_os.__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]) if config else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Create default app instance
app = create_app()


# Dependency injection
def get_client() -> RAGClient:
    """Get the RAG client instance."""
    if state.client is None:
        state.client = RAGClient()
    return state.client


def get_rate_limiter() -> RateLimiter:
    """Get the rate limiter instance."""
    if state.rate_limiter is None:
        state.rate_limiter = RateLimiter()
    return state.rate_limiter


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    state.request_count += 1
    request_id = str(uuid4())
    request.state.request_id = request_id

    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds() * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round(duration, 2))

    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


# Health endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version=rag_os.__version__,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
    )


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint."""
    if state.client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check endpoint."""
    return {"status": "alive"}


# Query endpoints
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: Request,
    body: QueryRequest,
    client: RAGClient = Depends(get_client),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Execute a RAG query."""
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    rate_result = rate_limiter.check(key=client_ip)
    if not rate_result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {rate_result.retry_after:.1f}s",
        )

    # Input validation
    validation = validate_query(body.query)
    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query: {', '.join(validation.errors)}",
        )

    # Execute query
    start_time = datetime.now()
    try:
        result = client.query(
            body.query,
            pipeline=body.pipeline,
            context=body.context,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    duration = (datetime.now() - start_time).total_seconds() * 1000

    return QueryResponse(
        id=getattr(request.state, "request_id", str(uuid4())),
        query=body.query,
        result=result.data if hasattr(result, "data") else result,
        pipeline=body.pipeline,
        success=result.success if hasattr(result, "success") else True,
        duration_ms=duration,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/query/{query_id}", tags=["Query"])
async def get_query_result(query_id: str):
    """Get the result of a previous query (for async queries)."""
    # Placeholder for async query result retrieval
    raise HTTPException(status_code=404, detail="Query not found")


# Pipeline endpoints
@app.get("/pipelines", response_model=list[str], tags=["Pipelines"])
async def list_pipelines(client: RAGClient = Depends(get_client)):
    """List all available pipelines."""
    return client.get_pipelines()


@app.get("/pipelines/{name}", response_model=PipelineInfo, tags=["Pipelines"])
async def get_pipeline(name: str, client: RAGClient = Depends(get_client)):
    """Get information about a specific pipeline."""
    info = client.get_pipeline_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")

    return PipelineInfo(
        name=name,
        version=info.get("version", "unknown"),
        description=info.get("description", ""),
        steps=info.get("steps", []),
    )


@app.post("/pipelines", tags=["Pipelines"])
async def create_pipeline(
    spec: dict[str, Any],
    client: RAGClient = Depends(get_client),
):
    """Create a new pipeline from a specification."""
    try:
        from rag_os.core.spec import PipelineSpec
        pipeline_spec = PipelineSpec(**spec)
        client.load_pipeline(pipeline_spec)
        return {"status": "created", "name": pipeline_spec.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/pipelines/{name}", tags=["Pipelines"])
async def delete_pipeline(name: str, client: RAGClient = Depends(get_client)):
    """Delete a pipeline."""
    try:
        client.unload_pipeline(name)
        return {"status": "deleted", "name": name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# Index endpoints
@app.get("/indexes", tags=["Indexes"])
async def list_indexes():
    """List all available indexes."""
    # Placeholder
    return []


@app.post("/indexes", tags=["Indexes"])
async def create_index(body: IndexRequest, background_tasks: BackgroundTasks):
    """Create a new index."""
    index_id = str(uuid4())

    # Add background task for indexing
    async def index_documents():
        # Actual indexing logic would go here
        pass

    background_tasks.add_task(index_documents)

    return {
        "id": index_id,
        "name": body.name,
        "status": "indexing",
        "document_count": len(body.documents),
    }


@app.get("/indexes/{name}", tags=["Indexes"])
async def get_index(name: str):
    """Get information about an index."""
    raise HTTPException(status_code=404, detail=f"Index '{name}' not found")


@app.delete("/indexes/{name}", tags=["Indexes"])
async def delete_index(name: str):
    """Delete an index."""
    raise HTTPException(status_code=404, detail=f"Index '{name}' not found")


# Document endpoints
@app.post("/indexes/{index_name}/documents", tags=["Documents"])
async def add_document(
    index_name: str,
    body: DocumentRequest,
    background_tasks: BackgroundTasks,
):
    """Add a document to an index."""
    doc_id = str(uuid4())

    return {
        "id": doc_id,
        "index": index_name,
        "status": "indexed",
    }


@app.get("/indexes/{index_name}/documents/{doc_id}", tags=["Documents"])
async def get_document(index_name: str, doc_id: str):
    """Get a document from an index."""
    raise HTTPException(status_code=404, detail="Document not found")


@app.delete("/indexes/{index_name}/documents/{doc_id}", tags=["Documents"])
async def delete_document(index_name: str, doc_id: str):
    """Delete a document from an index."""
    raise HTTPException(status_code=404, detail="Document not found")


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return {
        "requests_total": state.request_count,
        "uptime_seconds": uptime,
        "version": rag_os.__version__,
    }


# Info endpoint
@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "RAG OS API",
        "version": rag_os.__version__,
        "docs": "/docs",
        "health": "/health",
    }
