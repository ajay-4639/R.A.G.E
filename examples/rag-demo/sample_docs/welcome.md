# Welcome to R.A.G.E

**R.A.G.E** (Retrieval-Augmented Generation Engine) is a fully customizable RAG framework. It provides a pipeline-based architecture for building RAG applications.

## Key Features

- **Pipeline Architecture**: Define RAG pipelines as composable steps (ingest, chunk, embed, retrieve, generate)
- **Multiple Providers**: Support for OpenAI, Anthropic, Cohere, and local models
- **Vector Storage**: Integrates with Qdrant, with support for in-memory indexes for testing
- **Document Types**: Ingest text files, Markdown, PDF, JSON, and CSV
- **Chunking Strategies**: Fixed-size, sentence-aware, token-aware, and recursive chunking
- **Retrieval Methods**: Vector similarity, hybrid search, multi-query, and MMR

## How It Works

1. **Upload Documents**: Upload your files through the web interface or API
2. **Automatic Processing**: Documents are chunked, embedded, and stored in Qdrant
3. **Ask Questions**: Query your documents using natural language
4. **Get Answers**: The system retrieves relevant context and generates accurate answers

## Architecture

R.A.G.E uses a step-based architecture where each step in the pipeline is a composable unit:

- **Ingestion Step**: Reads documents from files, URLs, or APIs
- **Chunking Step**: Splits documents into manageable chunks
- **Embedding Step**: Converts text chunks into vector embeddings
- **Indexing Step**: Stores embeddings in a vector database (Qdrant)
- **Retrieval Step**: Finds relevant chunks for a given query
- **Prompt Assembly Step**: Builds the prompt with context and instructions
- **LLM Step**: Generates the final answer using an LLM

## Configuration

The demo uses the following defaults:
- Embedding model: text-embedding-3-small (1536 dimensions)
- LLM model: gpt-4o-mini
- Chunk size: 500 characters with 50 character overlap
- Retrieval: Top 5 most relevant chunks
- Vector metric: Cosine similarity
