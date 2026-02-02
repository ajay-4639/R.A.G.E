"""
POC: Using RAG OS as a Remote Service (Docker-hosted)
=====================================================

This demonstrates how to use RAG OS from an external application
by connecting to a running RAG OS server via the RemoteClient.

Setup:
    1. Start RAG OS server:
       docker compose up -d rag-os-api

    2. Install the client in your project:
       pip install rag-os[remote]

    3. Run this script:
       python poc_remote_client.py
"""

from rag_os import RemoteClient, PipelineBuilder, StepBuilder
from rag_os.core.types import StepType


def main():
    # =========================================================
    # 1. Connect to the RAG OS server
    # =========================================================
    client = RemoteClient("http://localhost:8000")

    # Wait for server to be ready (useful in CI/CD or startup)
    if not client.wait_until_ready(timeout=10):
        print("Server is not available. Start it with: docker compose up -d rag-os-api")
        return

    health = client.health()
    print(f"Connected to RAG OS v{health.version} (uptime: {health.uptime_seconds:.0f}s)")

    # =========================================================
    # 2. Build a pipeline spec using the Python SDK
    # =========================================================
    pipeline = (
        PipelineBuilder("document-qa")
        .with_version("1.0.0")
        .with_description("Q&A pipeline for document retrieval")
        .add_ingestion_step(
            "ingest",
            step_class="TextFileIngestion",
            source_path="./docs",
            file_patterns=["*.txt", "*.md"],
        )
        .add_chunking_step(
            "chunk",
            step_class="RecursiveChunking",
            depends_on="ingest",
            chunk_size=512,
            chunk_overlap=50,
        )
        .add_embedding_step(
            "embed",
            step_class="OpenAIEmbedding",
            depends_on="chunk",
            model="text-embedding-3-small",
        )
        .add_retrieval_step(
            "retrieve",
            step_class="VectorRetrieval",
            depends_on="embed",
            top_k=5,
        )
        .add_step(
            StepBuilder("rerank")
            .with_class("CrossEncoderReranking")
            .with_type(StepType.RERANKING)
            .depends_on("retrieve")
            .with_config(top_k=3)
        )
        .add_step(
            StepBuilder("prompt")
            .with_class("RAGTemplate")
            .with_type(StepType.PROMPT_ASSEMBLY)
            .depends_on("rerank")
            .with_config(
                template="Context: {context}\n\nQuestion: {query}\n\nAnswer:",
                max_context_tokens=2000,
            )
        )
        .add_llm_step(
            "generate",
            step_class="OpenAILLM",
            depends_on="prompt",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
        )
        .build()
    )

    # =========================================================
    # 3. Deploy the pipeline to the server
    # =========================================================
    name = client.deploy_pipeline(pipeline)
    print(f"Deployed pipeline: {name}")

    # List all deployed pipelines
    pipelines = client.list_pipelines()
    print(f"Active pipelines: {pipelines}")

    # =========================================================
    # 4. Query the pipeline
    # =========================================================
    result = client.query(
        "What is the return policy for electronics?",
        pipeline="document-qa",
    )

    print(f"\nQuery: {result.query}")
    print(f"Answer: {result.answer}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_ms:.0f}ms")

    if result.sources:
        print(f"\nSources ({len(result.sources)}):")
        for src in result.sources:
            print(f"  - {src.get('content', '')[:100]}...")

    # =========================================================
    # 5. Get pipeline details
    # =========================================================
    info = client.get_pipeline("document-qa")
    print(f"\nPipeline: {info.name} v{info.version}")
    print(f"Steps: {len(info.steps)}")
    for step in info.steps:
        print(f"  - {step['id']} ({step.get('step_class', '')})")

    # =========================================================
    # 6. Check metrics
    # =========================================================
    metrics = client.metrics()
    print(f"\nServer metrics: {metrics}")

    # =========================================================
    # 7. Cleanup (optional)
    # =========================================================
    # client.delete_pipeline("document-qa")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
