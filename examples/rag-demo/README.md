# R.A.G.E Demo

A complete RAG application with document upload, vector search, and chat interface.

**R.A.G.E** = Retrieval-Augmented Generation Engine

## Prerequisites

- Docker containers running (Qdrant): `docker compose up -d`
- Python 3.10+
- OpenAI API key

## Setup

```bash
# 1. Navigate to demo directory
cd examples/rag-demo

# 2. Copy environment template and add your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here

# 3. Install R.A.G.E from source (if not already installed)
pip install -e ../../

# 4. Install demo dependencies
pip install -r requirements.txt

# 5. Run the demo
python -m uvicorn app:app --port 8080 --reload
```

Open http://localhost:8080

## Usage

1. **Upload documents** using the sidebar (drag-and-drop or click). Supports `.txt`, `.md`, `.pdf`
2. **Ask questions** in the chat. The system retrieves relevant context from your documents and generates answers
3. **View sources** by clicking the sources toggle under each answer

## Architecture

```
Browser (port 8080)
    |
    v
Demo FastAPI App
    |
    |-- OpenAIEmbeddingStep (text-embedding-3-small)
    |-- RecursiveChunkingStep (500 chars)
    |-- RAGPromptAssemblyStep
    |-- OpenAILLMStep (gpt-4o-mini)
    |
    v
Qdrant (port 6333) -- vector storage
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/api/upload` | Upload a document |
| GET | `/api/documents` | List documents |
| DELETE | `/api/documents/{id}` | Delete a document |
| POST | `/api/query` | Query the RAG pipeline |
| GET | `/api/stats` | Index statistics |
| GET | `/api/health` | Health check |
