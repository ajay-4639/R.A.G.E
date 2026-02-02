# RAG OS POC - Example Integration

A complete working example showing how to integrate RAG OS into your application.

## Quick Start

### 1. Add your API key

```bash
cp .env.example .env
```

Edit `.env` and add **one** of these:
```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-key-here
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run it

**Local mode** (no Docker needed):
```bash
python app.py
```

**Remote mode** (with Docker):
```bash
# Terminal 1: Start RAG OS server
docker compose up -d rag-os-api

# Terminal 2: Run the app
python app.py remote
```

## What's Included

| File | Description |
|------|-------------|
| `app.py` | Interactive CLI chatbot - ask questions about ACME Corp docs |
| `web_app.py` | FastAPI web app integration - shows how to embed RAG OS in your API |
| `docs/` | Sample documents (company policies, FAQ, technical docs) |
| `.env.example` | Environment variables template |

## How It Works

```
Your App                    RAG OS (Docker or Library)
--------                    -------------------------

  User asks               1. Ingest docs from docs/
  a question    ------>   2. Chunk into 500-char pieces
                          3. Create embeddings (OpenAI/Gemini)
  Gets back     <------   4. Retrieve top 5 relevant chunks
  an answer               5. Assemble prompt with context
                          6. Generate answer via LLM
```

## Two Integration Modes

### Library Mode (`from rag_os import RAGClient`)
Everything runs in your process. No Docker needed.

```python
from rag_os import RAGClient, PipelineBuilder

client = RAGClient()
pipeline = PipelineBuilder("my-rag").add_llm_step(...).build()
client.load_pipeline(pipeline)
result = client.query("What is the return policy?", pipeline="my-rag")
print(result["answer"])
```

### Service Mode (`from rag_os import RemoteClient`)
RAG OS runs as a Docker container. Your app connects via HTTP.

```python
from rag_os import RemoteClient

client = RemoteClient("http://localhost:8000")
client.deploy_pipeline(pipeline_spec)
result = client.query("What is the return policy?", pipeline="my-rag")
print(result.answer)
```

## Sample Questions

Try asking these:
- "What is the return policy for electronics?"
- "How many days of PTO do employees get?"
- "What payment methods do you accept?"
- "What is the API rate limit for the pro tier?"
- "How do I contact customer support?"
