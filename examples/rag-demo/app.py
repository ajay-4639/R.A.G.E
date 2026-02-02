"""R.A.G.E Demo - FastAPI Web Application.

A simple web app demonstrating R.A.G.E (Retrieval-Augmented Generation Engine) integration with:
- Document upload (txt, md, pdf)
- Chat-based querying with source citations
- Qdrant vector storage
- OpenAI embeddings + LLM

Usage:
    cp .env.example .env  # add your OPENAI_API_KEY
    pip install -r requirements.txt
    python -m uvicorn app:app --port 8080 --reload
"""

import os
import sys
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_engine import RAGEngine

# ---- App Setup ----

app = FastAPI(
    title="R.A.G.E Demo",
    description="Document Q&A powered by R.A.G.E (Retrieval-Augmented Generation Engine)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Global engine instance
engine: RAGEngine | None = None

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---- Request Models ----

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ---- Lifecycle ----

@app.on_event("startup")
async def startup():
    global engine

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set. The demo will not work without it. "
            "Set it in your .env file."
        )
        return

    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
    collection = os.environ.get("COLLECTION_NAME", "rag_demo")

    try:
        engine = RAGEngine(
            openai_api_key=api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection,
        )
        logger.info(f"RAG Engine initialized (Qdrant: {qdrant_host}:{qdrant_port}, collection: {collection})")

        # Auto-ingest sample docs if they exist
        sample_dir = BASE_DIR / "sample_docs"
        if sample_dir.exists():
            for f in sorted(sample_dir.iterdir()):
                if f.suffix in ALLOWED_EXTENSIONS:
                    try:
                        result = engine.ingest_path(str(f))
                        logger.info(f"Auto-ingested {f.name}: {result['chunk_count']} chunks")
                    except Exception as e:
                        logger.warning(f"Could not auto-ingest {f.name}: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        engine = None


def _require_engine() -> RAGEngine:
    """Get the engine or raise 503."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Engine not initialized. Check OPENAI_API_KEY and Qdrant connection.",
        )
    return engine


# ---- Page Routes ----

@app.get("/")
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---- API Routes ----

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and ingest a document."""
    eng = _require_engine()

    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        result = eng.ingest_file(content, file.filename or "document.txt")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    eng = _require_engine()
    return eng.list_documents()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks."""
    eng = _require_engine()
    deleted = eng.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/api/query")
async def query(body: QueryRequest):
    """Query the RAG pipeline."""
    eng = _require_engine()

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = eng.query(body.query, top_k=body.top_k)
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get index statistics."""
    eng = _require_engine()
    return eng.get_stats()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_initialized", "detail": "Check OPENAI_API_KEY and Qdrant"},
        )
    return engine.health_check()
