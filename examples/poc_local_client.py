"""
POC: Using RAG OS as a Local Library (pip install)
===================================================

This demonstrates how to use RAG OS directly in your Python code
without running a separate server. Just pip install and import.

Setup:
    pip install rag-os

    # Or install from local source:
    pip install -e .

Usage:
    python poc_local_client.py
"""

from rag_os import RAGClient, PipelineBuilder, PipelineSpec
from rag_os.core.types import StepType
from rag_os.sdk.builder import StepBuilder


def main():
    # =========================================================
    # 1. Create a RAG client (runs locally, no server needed)
    # =========================================================
    client = RAGClient()

    # =========================================================
    # 2. Build a pipeline using the fluent builder API
    # =========================================================
    pipeline = (
        PipelineBuilder("local-qa")
        .with_version("1.0.0")
        .with_description("Local Q&A pipeline")
        .add_ingestion_step(
            "ingest",
            step_class="TextFileIngestion",
            source_path="./docs",
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
        .add_llm_step(
            "generate",
            step_class="OpenAILLM",
            depends_on="retrieve",
            model="gpt-4o-mini",
            temperature=0.7,
        )
        .build()
    )

    # =========================================================
    # 3. Load pipeline into the client
    # =========================================================
    name = client.load_pipeline(pipeline)
    print(f"Loaded pipeline: {name}")

    # =========================================================
    # 4. Query the pipeline
    # =========================================================
    result = client.query("What is RAG?", pipeline="local-qa")
    print(f"Answer: {result.get('answer', 'N/A')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Latency: {result.get('latency_ms', 0):.0f}ms")

    # =========================================================
    # 5. Load a pipeline from a file
    # =========================================================
    # From JSON:
    # client.load_pipeline("my-pipeline.json")

    # From YAML:
    # client.load_pipeline("my-pipeline.yaml")

    # From a dict (e.g., API response or config):
    # client.load_pipeline({
    #     "name": "from-dict",
    #     "version": "1.0.0",
    #     "steps": [...]
    # })

    # =========================================================
    # 6. Export pipeline spec for deployment
    # =========================================================
    # Save to file for later deployment to Docker
    pipeline.to_json("./my-pipeline.json")
    pipeline.to_yaml("./my-pipeline.yaml")
    print("\nExported pipeline specs to JSON and YAML")

    # =========================================================
    # 7. List and manage pipelines
    # =========================================================
    print(f"Loaded pipelines: {client.list_pipelines()}")

    info = client.get_pipeline_info("local-qa")
    print(f"Pipeline info: {info}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
