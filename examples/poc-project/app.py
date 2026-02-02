"""
RAG OS POC - Complete Working Example
======================================

A complete RAG application that:
1. Reads documents from the docs/ folder
2. Chunks them into smaller pieces
3. Creates embeddings using OpenAI/Anthropic/Gemini
4. Stores them for retrieval
5. Answers questions using the retrieved context

Usage:
    1. Copy .env.example to .env and add your API key
    2. pip install -r requirements.txt
    3. python app.py

Two modes:
    - LOCAL:  Runs everything in-process (no Docker needed)
    - REMOTE: Connects to a running RAG OS Docker container
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_provider():
    """Detect which LLM provider is configured."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.getenv("GOOGLE_API_KEY"):
        return "google"
    else:
        print("ERROR: No API key found!")
        print("Copy .env.example to .env and add your API key.")
        print("")
        print("Supported providers:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  GOOGLE_API_KEY=...")
        sys.exit(1)


def get_llm_config(provider: str) -> dict:
    """Get LLM step configuration for the detected provider."""
    configs = {
        "openai": {
            "step_class": "OpenAILLM",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "anthropic": {
            "step_class": "AnthropicLLM",
            "model": "claude-3-haiku-20240307",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "google": {
            "step_class": "GeminiLLM",
            "model": "gemini-1.5-flash",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    }
    return configs[provider]


def get_embedding_config(provider: str) -> dict:
    """Get embedding step configuration for the detected provider."""
    configs = {
        "openai": {
            "step_class": "OpenAIEmbedding",
            "model": "text-embedding-3-small",
        },
        "anthropic": {
            "step_class": "OpenAIEmbedding",  # Anthropic doesn't have embeddings, use OpenAI
            "model": "text-embedding-3-small",
        },
        "google": {
            "step_class": "GeminiEmbedding",
            "model": "text-embedding-004",
        },
    }
    return configs[provider]


def build_pipeline(provider: str):
    """Build the RAG pipeline spec."""
    from rag_os import PipelineBuilder
    from rag_os.sdk.builder import StepBuilder
    from rag_os.core.types import StepType

    llm_config = get_llm_config(provider)
    embed_config = get_embedding_config(provider)

    docs_path = str(Path(__file__).parent / "docs")

    pipeline = (
        PipelineBuilder("acme-qa")
        .with_version("1.0.0")
        .with_description(f"ACME Corp Q&A pipeline using {provider}")

        # Step 1: Ingest documents from the docs/ folder
        .add_ingestion_step(
            "ingest",
            step_class="TextFileIngestion",
            source_path=docs_path,
            file_patterns=["*.txt", "*.md"],
        )

        # Step 2: Chunk documents into smaller pieces
        .add_chunking_step(
            "chunk",
            step_class="RecursiveChunking",
            depends_on="ingest",
            chunk_size=500,
            chunk_overlap=50,
        )

        # Step 3: Create embeddings
        .add_embedding_step(
            "embed",
            depends_on="chunk",
            **embed_config,
        )

        # Step 4: Retrieve relevant chunks
        .add_retrieval_step(
            "retrieve",
            step_class="VectorRetrieval",
            depends_on="embed",
            top_k=5,
            min_score=0.3,
        )

        # Step 5: Assemble the prompt with retrieved context
        .add_step(
            StepBuilder("prompt")
            .with_class("RAGTemplate")
            .with_type(StepType.PROMPT_ASSEMBLY)
            .depends_on("retrieve")
            .with_config(
                template=(
                    "You are a helpful assistant for ACME Corp. "
                    "Use the following context to answer the question. "
                    "If the answer is not in the context, say so.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {query}\n\n"
                    "Answer:"
                ),
                max_context_tokens=2000,
            )
        )

        # Step 6: Generate answer using LLM
        .add_llm_step(
            "generate",
            depends_on="prompt",
            **llm_config,
        )

        .build()
    )

    return pipeline


def run_local_mode(provider: str):
    """Run RAG OS directly in-process (no Docker needed)."""
    from rag_os import RAGClient

    print(f"Running in LOCAL mode with {provider}")
    print("=" * 50)

    # Build and load pipeline
    pipeline = build_pipeline(provider)
    client = RAGClient()
    client.load_pipeline(pipeline)
    print(f"Pipeline '{pipeline.name}' loaded with {len(pipeline.steps)} steps\n")

    # Interactive query loop
    print("Ask questions about ACME Corp (type 'quit' to exit):\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = client.query(query, pipeline="acme-qa")
        print(f"\nAssistant: {result.get('answer', 'No answer')}")
        print(f"  (latency: {result.get('latency_ms', 0):.0f}ms)\n")

    client.close()


def run_remote_mode(provider: str):
    """Run RAG OS via Docker-hosted server."""
    from rag_os import RemoteClient

    server_url = os.getenv("RAG_OS_SERVER", "http://localhost:8000")
    print(f"Running in REMOTE mode -> {server_url}")
    print(f"Using provider: {provider}")
    print("=" * 50)

    client = RemoteClient(server_url)

    # Check server health
    if not client.is_healthy():
        print(f"\nCannot connect to RAG OS server at {server_url}")
        print("Start it with: docker compose up -d rag-os-api")
        return

    health = client.health()
    print(f"Server: RAG OS v{health.version} (uptime: {health.uptime_seconds:.0f}s)")

    # Deploy pipeline
    pipeline = build_pipeline(provider)
    name = client.deploy_pipeline(pipeline)
    print(f"Deployed pipeline: {name}\n")

    # Interactive query loop
    print("Ask questions about ACME Corp (type 'quit' to exit):\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = client.query(query, pipeline="acme-qa")
        print(f"\nAssistant: {result.answer}")
        print(f"  (latency: {result.duration_ms:.0f}ms)\n")

        if result.sources:
            print(f"  Sources ({len(result.sources)}):")
            for src in result.sources:
                print(f"    - {src.get('content', '')[:80]}...")
            print()

    client.close()


def main():
    provider = get_provider()
    print(f"\nRAG OS POC - ACME Corp Q&A")
    print(f"Provider: {provider.upper()}\n")

    # Check which mode to run
    mode = os.getenv("RAG_OS_MODE", "local").lower()

    # If RAG_OS_SERVER is set, default to remote mode
    if os.getenv("RAG_OS_SERVER"):
        mode = "remote"

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "remote":
        run_remote_mode(provider)
    else:
        run_local_mode(provider)


if __name__ == "__main__":
    main()
