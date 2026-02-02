"""
RAG OS POC - Web App Integration Example
==========================================

Shows how to integrate RAG OS into an existing FastAPI/Flask web application.
Your app stays independent -- RAG OS is just a service you call.

Usage:
    1. Start RAG OS:       docker compose up -d rag-os-api
    2. Add your key:       cp .env.example .env && edit .env
    3. Run this app:       python web_app.py
    4. Open browser:       http://localhost:3000/docs

This is how you'd integrate RAG OS into a real product:
    - Your app has its own routes, database, auth, etc.
    - RAG OS handles the RAG pipeline as a backend service.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# =========================================================
# Your application (this is YOUR code, not RAG OS)
# =========================================================
app = FastAPI(
    title="ACME Corp Support Bot",
    description="Customer support powered by RAG OS",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# =========================================================
# RAG OS integration layer
# =========================================================
from rag_os import RemoteClient, PipelineBuilder
from rag_os.sdk.builder import StepBuilder
from rag_os.core.types import StepType

# Connect to the RAG OS server (Docker container)
rag = RemoteClient(os.getenv("RAG_OS_SERVER", "http://localhost:8000"))


def setup_rag_pipeline():
    """Deploy the support pipeline to RAG OS on app startup."""
    docs_path = str(Path(__file__).parent / "docs")

    pipeline = (
        PipelineBuilder("support-bot")
        .with_version("1.0.0")
        .with_description("Customer support Q&A pipeline")
        .add_ingestion_step("ingest", source_path=docs_path)
        .add_chunking_step("chunk", depends_on="ingest", chunk_size=500, chunk_overlap=50)
        .add_embedding_step("embed", depends_on="chunk", model="text-embedding-3-small")
        .add_retrieval_step("retrieve", depends_on="embed", top_k=5)
        .add_step(
            StepBuilder("prompt")
            .with_class("RAGTemplate")
            .with_type(StepType.PROMPT_ASSEMBLY)
            .depends_on("retrieve")
            .with_config(
                template=(
                    "You are a customer support agent for ACME Corp. "
                    "Be friendly and helpful. Use the context below to answer.\n\n"
                    "Context:\n{context}\n\n"
                    "Customer question: {query}\n\n"
                    "Your response:"
                ),
            )
        )
        .add_llm_step("generate", depends_on="prompt", model="gpt-4o-mini", temperature=0.5)
        .build()
    )

    try:
        rag.deploy_pipeline(pipeline)
        print("RAG pipeline deployed successfully")
    except Exception as e:
        print(f"Warning: Could not deploy pipeline: {e}")
        print("Make sure RAG OS server is running: docker compose up -d rag-os-api")


# =========================================================
# Your app's API routes
# =========================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict] = []
    response_time_ms: float = 0


@app.get("/")
async def home():
    """Your app's homepage."""
    return {
        "app": "ACME Corp Support Bot",
        "status": "running",
        "rag_status": "healthy" if rag.is_healthy() else "unavailable",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Handle customer support chat messages.

    This is YOUR endpoint. It calls RAG OS internally to get answers.
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    try:
        # Call RAG OS to get an answer
        result = rag.query(req.message, pipeline="support-bot")

        return ChatResponse(
            reply=result.answer,
            sources=result.sources,
            response_time_ms=result.duration_ms,
        )
    except Exception as e:
        # Fallback if RAG OS is unavailable
        return ChatResponse(
            reply="I'm sorry, I'm having trouble right now. Please contact support@acmecorp.com.",
            sources=[],
            response_time_ms=0,
        )


@app.get("/health")
async def health():
    """Health check for your app + RAG OS."""
    rag_healthy = rag.is_healthy()
    return {
        "app": "healthy",
        "rag_os": "healthy" if rag_healthy else "unavailable",
        "pipeline": "support-bot",
    }


@app.get("/pipelines")
async def list_pipelines():
    """Show deployed RAG pipelines."""
    try:
        return {"pipelines": rag.list_pipelines()}
    except Exception:
        return {"pipelines": [], "error": "RAG OS unavailable"}


# =========================================================
# Startup
# =========================================================
@app.on_event("startup")
async def on_startup():
    setup_rag_pipeline()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
