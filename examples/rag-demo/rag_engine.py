"""R.A.G.E Engine - Orchestrates R.A.G.E steps for the demo application.

R.A.G.E = Retrieval-Augmented Generation Engine

Wires up all step classes (embedding, chunking, indexing,
prompt assembly, LLM) and provides a simple API for document
ingestion and querying.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

# Add parent to path so we can import rag_os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_os.models.document import Document, SourceType
from rag_os.models.chunk import Chunk
from rag_os.models.index import IndexConfig, DistanceMetric
from rag_os.steps.embedding.providers import OpenAIEmbeddingStep
from rag_os.steps.chunking.strategies import RecursiveChunkingStep
from rag_os.steps.indexing.qdrant_index import QdrantVectorIndex
from rag_os.steps.prompt_assembly.assemblers import RAGPromptAssemblyStep
from rag_os.steps.llm.providers import OpenAILLMStep
from rag_os.steps.ingestion.file_ingestors import TextFileIngestionStep, MarkdownIngestionStep
from rag_os.steps.ingestion.pdf_ingestor import PDFIngestionStep


class RAGEngine:
    """Orchestrates the full RAG pipeline using rag_os components.

    This engine directly instantiates rag_os step classes and calls
    their methods, rather than going through the PipelineExecutor.
    This gives us full control over wiring shared state (like the
    vector index and embedder) between steps.
    """

    def __init__(
        self,
        openai_api_key: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "rag_demo",
    ):
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name

        # Initialize embedding step
        self.embedder = OpenAIEmbeddingStep(
            step_id="embedder",
            config={
                "api_key": openai_api_key,
                "model_name": "text-embedding-3-small",
            },
        )

        # Initialize Qdrant vector index
        index_config = IndexConfig(
            name=collection_name,
            dimensions=1536,  # text-embedding-3-small dimensions
            metric=DistanceMetric.COSINE,
        )
        self.index = QdrantVectorIndex(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
            config=index_config,
        )

        # Initialize chunker
        self.chunker = RecursiveChunkingStep(
            step_id="chunker",
            config={
                "chunk_size": 500,
                "chunk_overlap": 50,
                "min_chunk_size": 50,
            },
        )

        # Initialize prompt assembler
        self.prompt_assembler = RAGPromptAssemblyStep(
            step_id="prompt",
            config={
                "system_prompt": (
                    "You are a helpful assistant that answers questions "
                    "based on the provided documents. Be accurate and concise."
                ),
                "cite_sources": True,
            },
        )

        # Initialize LLM
        self.llm = OpenAILLMStep(
            step_id="llm",
            config={
                "api_key": openai_api_key,
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 1000,
            },
        )

        # In-memory document metadata store
        self._documents: dict[str, dict[str, Any]] = {}

        # Ingestors (lazy-initialized per type)
        self._ingestors: dict[str, Any] = {}

    def _get_ingestor(self, file_type: str) -> Any:
        """Get or create an ingestor for the given file type."""
        if file_type not in self._ingestors:
            config = {"extract_metadata": True, "skip_errors": False}
            if file_type == "txt":
                self._ingestors[file_type] = TextFileIngestionStep(
                    step_id=f"ingest_{file_type}", config=config
                )
            elif file_type == "md":
                self._ingestors[file_type] = MarkdownIngestionStep(
                    step_id=f"ingest_{file_type}", config=config
                )
            elif file_type == "pdf":
                self._ingestors[file_type] = PDFIngestionStep(
                    step_id=f"ingest_{file_type}", config=config
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        return self._ingestors[file_type]

    def ingest_file(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """Ingest a file into the RAG pipeline.

        Steps: detect type -> ingest -> chunk -> embed -> store in Qdrant

        Args:
            file_bytes: Raw file content
            filename: Original filename (used for type detection)

        Returns:
            Dict with doc_id, filename, chunk_count, etc.
        """
        start_time = time.time()

        # Detect file type
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in ("txt", "md", "pdf"):
            raise ValueError(f"Unsupported file type: .{suffix}. Use .txt, .md, or .pdf")

        # Step 1: Ingest - convert file to Document
        ingestor = self._get_ingestor(suffix)

        if suffix == "pdf":
            documents = ingestor.ingest({
                "file_bytes": file_bytes,
                "file_name": filename,
            })
        else:
            # For text/md files, write to temp file and ingest
            with tempfile.NamedTemporaryFile(
                suffix=f".{suffix}", delete=False, mode="wb"
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                documents = ingestor.ingest({"source_path": tmp_path})
                # Fix the title to use original filename
                for doc in documents:
                    doc.title = Path(filename).stem
            finally:
                os.unlink(tmp_path)

        if not documents:
            raise ValueError("No content could be extracted from the file")

        doc = documents[0]

        # Step 2: Chunk the document
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            raise ValueError("Document produced no chunks")

        # Step 3: Embed the chunks
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # Step 4: Store in Qdrant
        count = self.index.add(embedded_chunks)

        # Track document metadata
        duration_ms = (time.time() - start_time) * 1000
        doc_info = {
            "doc_id": doc.doc_id,
            "filename": filename,
            "title": doc.title,
            "chunk_count": count,
            "char_count": len(doc.content),
            "file_type": suffix,
            "ingested_at": time.time(),
            "duration_ms": round(duration_ms, 1),
        }
        self._documents[doc.doc_id] = doc_info

        return doc_info

    def ingest_path(self, file_path: str) -> dict[str, Any]:
        """Ingest a file from a local path.

        Args:
            file_path: Path to the file

        Returns:
            Dict with doc_id, filename, chunk_count, etc.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_bytes = path.read_bytes()
        return self.ingest_file(file_bytes, path.name)

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query the RAG pipeline.

        Steps: embed query -> search Qdrant -> assemble prompt -> generate

        Args:
            question: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            Dict with answer, sources, duration_ms
        """
        start_time = time.time()

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Search Qdrant
        results = self.index.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": "I don't have any documents to answer from. Please upload some documents first.",
                "sources": [],
                "duration_ms": round((time.time() - start_time) * 1000, 1),
            }

        # Step 3: Assemble prompt
        assembled = self.prompt_assembler.assemble(question, results)

        # Step 4: Generate answer
        llm_response = self.llm.generate(assembled)

        duration_ms = (time.time() - start_time) * 1000

        # Format sources
        sources = []
        for i, result in enumerate(results):
            sources.append({
                "index": i + 1,
                "content": result.content[:300],
                "score": round(result.score, 4),
                "doc_id": result.doc_id,
                "filename": self._documents.get(result.doc_id, {}).get("filename", "Unknown"),
            })

        return {
            "answer": llm_response.content,
            "sources": sources,
            "duration_ms": round(duration_ms, 1),
            "tokens": {
                "prompt": llm_response.prompt_tokens,
                "completion": llm_response.completion_tokens,
                "total": llm_response.total_tokens,
            },
        }

    def list_documents(self) -> list[dict[str, Any]]:
        """List all ingested documents."""
        return list(self._documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from Qdrant.

        Args:
            doc_id: The document ID to delete

        Returns:
            True if the document was found and deleted
        """
        if doc_id not in self._documents:
            return False

        self.index.delete_by_doc_id(doc_id)
        del self._documents[doc_id]
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the index and documents."""
        stats = self.index.get_stats()
        return {
            "total_documents": len(self._documents),
            "total_chunks": stats.total_chunks,
            "dimensions": stats.dimensions,
            "collection_name": self.collection_name,
        }

    def health_check(self) -> dict[str, Any]:
        """Check connectivity to Qdrant and OpenAI."""
        status = {"qdrant": "unknown", "openai": "unknown"}

        # Check Qdrant
        try:
            self.index.count()
            status["qdrant"] = "healthy"
        except Exception as e:
            status["qdrant"] = f"error: {e}"

        # Check OpenAI (lightweight - just verify API key format)
        if self.openai_api_key and self.openai_api_key.startswith("sk-"):
            status["openai"] = "configured"
        else:
            status["openai"] = "not configured"

        return status
