# R.A.G.E (RAG Engine)

A fully customizable RAG Operating System — a pipeline compiler and runtime for building production-grade Retrieval-Augmented Generation systems.

## Features

- **Modular Pipeline Architecture** — Compose pipelines from reusable steps
- **Multiple Step Types** — Ingestion, Chunking, Embedding, Indexing, Retrieval, Reranking, Prompt Assembly, LLM Execution, Post-Processing
- **Plugin System** — Extend functionality with custom plugins and hooks
- **Evaluation Framework** — Built-in metrics for retrieval recall, precision, faithfulness, and hallucination detection
- **Tracing & Observability** — Full trace collection with step-level metrics and token usage tracking
- **Security** — Input validation, rate limiting, secrets management, and PII detection
- **Multiple Storage Backends** — In-memory, file-based, and extensible storage interfaces
- **SDK & CLI** — Python SDK for programmatic access and CLI for command-line operations

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from rag_os.sdk import RAGSession, PipelineBuilder

# Build a pipeline
builder = PipelineBuilder("my-pipeline")
builder.add_step("ingestion", "text-file", config={"path": "docs/"})
builder.add_step("chunking", "token-aware", config={"chunk_size": 512})
builder.add_step("embedding", "openai", config={"model": "text-embedding-3-small"})
builder.add_step("indexing", "in-memory")
spec = builder.build()

# Run a query
session = RAGSession()
session.load_pipeline(spec)
result = session.run("What is RAG?")
print(result.output)
```

## Architecture

```
rag_os/
├── core/           # Pipeline execution engine
│   ├── step.py         # Base Step class
│   ├── spec.py         # Pipeline specification
│   ├── registry.py     # Step registry
│   ├── validator.py    # Pipeline validation
│   ├── executor.py     # Sync executor
│   └── async_executor.py  # Async/batch executor
├── models/         # Data models
│   ├── document.py     # Document model
│   ├── chunk.py        # Chunk model
│   ├── embedding.py    # Embedding model
│   └── index.py        # Index models
├── steps/          # Pipeline steps
│   ├── ingestion/      # File, URL, API ingestors
│   ├── chunking/       # Fixed, token-aware, semantic chunking
│   ├── embedding/      # OpenAI, local, Cohere providers
│   ├── indexing/       # Vector index implementations
│   ├── retrieval/      # Top-k, hybrid, multi-query retrieval
│   ├── reranking/      # Cross-encoder, LLM-based reranking
│   ├── prompt_assembly/  # Template-based prompt building
│   ├── llm/            # LLM provider integrations
│   └── post_processing/  # Output validation, PII masking
├── storage/        # Persistence layer
├── tracing/        # Observability
├── eval/           # Evaluation framework
├── plugins/        # Plugin system
├── security/       # Security features
├── sdk/            # Python SDK
├── cli/            # Command-line interface
└── api/            # REST API models
```

## CLI Usage

```bash
# Validate a pipeline
rag validate pipeline.yaml

# Run a query
rag run my-pipeline "What is machine learning?"

# List available pipelines
rag list-pipelines

# Run evaluation
rag eval my-pipeline eval_dataset.json
```

## Pipeline Specification

Pipelines are defined in YAML or JSON:

```yaml
name: my-rag-pipeline
version: "1.0.0"
steps:
  - step_id: ingest
    step_type: INGESTION
    config:
      source_type: pdf
      path: "./documents"

  - step_id: chunk
    step_type: CHUNKING
    config:
      strategy: token_aware
      chunk_size: 512
      overlap: 50

  - step_id: embed
    step_type: EMBEDDING
    config:
      provider: openai
      model: text-embedding-3-small

  - step_id: retrieve
    step_type: RETRIEVAL
    config:
      top_k: 10

  - step_id: generate
    step_type: LLM_EXECUTION
    config:
      provider: openai
      model: gpt-4
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test file
pytest tests/unit/test_core.py
```

## License

MIT
