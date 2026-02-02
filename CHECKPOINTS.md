# RAG OS — BUILD CHECKPOINTS

## Overview
This document contains the complete, ordered checkpoint list for building the RAG OS.
Each checkpoint is atomic, testable, and builds on the previous one.

---

# PHASE 1: FOUNDATION (Core Abstractions)

## Checkpoint 1.1: Project Structure & Base Classes
- [ ] Create project directory structure
- [ ] Set up Python package with `pyproject.toml`
- [ ] Create base `Step` abstract class with:
  - `step_id: str`
  - `step_type: StepType (enum)`
  - `input_schema: dict`
  - `output_schema: dict`
  - `config_schema: dict`
  - `validate_input()` method
  - `validate_output()` method
  - `execute(context: StepContext) -> StepResult` abstract method
- [ ] Create `StepContext` dataclass (carries data between steps)
- [ ] Create `StepResult` dataclass (output + metadata)
- [ ] Create `StepType` enum (INGESTION, CHUNKING, EMBEDDING, etc.)
- [ ] Write unit tests for base classes

**Deliverable:** Can instantiate a dummy step that passes validation

---

## Checkpoint 1.2: Pipeline Spec Schema
- [ ] Define JSON Schema for pipeline specification
- [ ] Create `PipelineSpec` Pydantic model with:
  - `name: str`
  - `version: str`
  - `steps: List[StepSpec]`
  - `metadata: dict`
- [ ] Create `StepSpec` Pydantic model with:
  - `step_id: str`
  - `step_type: StepType`
  - `config: dict`
  - `dependencies: List[str]` (for future DAG support)
- [ ] Implement `PipelineSpec.from_yaml()` loader
- [ ] Implement `PipelineSpec.from_json()` loader
- [ ] Implement schema validation on load
- [ ] Write sample pipeline YAML files
- [ ] Write unit tests for spec loading/validation

**Deliverable:** Can load and validate a pipeline spec from YAML/JSON

---

## Checkpoint 1.3: Step Registry & Discovery
- [ ] Create `StepRegistry` singleton class
- [ ] Implement `@register_step` decorator
- [ ] Implement `registry.get(step_type, step_id)` lookup
- [ ] Implement `registry.list_steps()` for discovery
- [ ] Add step metadata (description, author, version)
- [ ] Write tests for registration and lookup

**Deliverable:** Can register custom steps and look them up by type/id

---

## Checkpoint 1.4: Pipeline Validator
- [ ] Create `PipelineValidator` class
- [ ] Implement step existence validation (all steps in spec exist in registry)
- [ ] Implement input/output schema compatibility checks between adjacent steps
- [ ] Implement circular dependency detection (for future DAG)
- [ ] Implement config schema validation per step
- [ ] Return structured validation errors with line numbers
- [ ] Write tests with valid and invalid pipelines

**Deliverable:** Can validate a pipeline spec and get detailed errors

---

# PHASE 2: EXECUTION ENGINE

## Checkpoint 2.1: Runtime Context
- [ ] Create `RuntimeContext` class with:
  - `trace_id: str` (auto-generated UUID)
  - `user_metadata: dict`
  - `pipeline_version: str`
  - `token_budget: int`
  - `cost_budget: float`
  - `start_time: datetime`
- [ ] Implement context propagation helpers
- [ ] Implement `runtime_overrides: dict` support
- [ ] Add RBAC scope placeholder (for future)
- [ ] Write tests for context creation and propagation

**Deliverable:** Can create and propagate runtime context

---

## Checkpoint 2.2: Pipeline Executor (Sync)
- [ ] Create `PipelineExecutor` class
- [ ] Implement `execute(spec: PipelineSpec, context: RuntimeContext) -> PipelineResult`
- [ ] Implement linear step execution loop
- [ ] Implement step input/output validation at runtime
- [ ] Implement step timing/metrics collection
- [ ] Create `PipelineResult` with:
  - `success: bool`
  - `output: Any`
  - `step_results: List[StepResult]`
  - `total_latency_ms: float`
  - `trace_id: str`
- [ ] Write integration tests with mock steps

**Deliverable:** Can execute a multi-step pipeline synchronously

---

## Checkpoint 2.3: Error Handling & Recovery
- [ ] Define `StepError` exception hierarchy
- [ ] Implement step-level try/catch isolation
- [ ] Implement `RetryPolicy` dataclass (max_retries, backoff)
- [ ] Add retry logic per step based on config
- [ ] Implement `FailureMode` enum (FAIL_FAST, GRACEFUL)
- [ ] Implement graceful degradation (skip non-critical steps)
- [ ] Implement `fallback_step` support in StepSpec
- [ ] Write tests for retry and fallback scenarios

**Deliverable:** Pipeline handles step failures gracefully with retries/fallbacks

---

## Checkpoint 2.4: Async & Batch Execution
- [ ] Create `AsyncPipelineExecutor` class
- [ ] Implement `async execute()` method
- [ ] Implement `execute_batch(inputs: List)` for batch processing
- [ ] Implement concurrency controls (max_concurrent_steps)
- [ ] Implement `DryRunExecutor` (skips LLM calls, returns mock)
- [ ] Add execution mode to RuntimeContext
- [ ] Write async tests

**Deliverable:** Can execute pipelines async and in batch mode

---

# PHASE 3: INGESTION SYSTEM

## Checkpoint 3.1: Document Model
- [ ] Create `Document` dataclass:
  - `doc_id: str`
  - `content: str`
  - `source_type: SourceType`
  - `source_uri: str`
  - `metadata: dict`
  - `acl: AccessControl`
  - `created_at: datetime`
  - `expires_at: Optional[datetime]`
- [ ] Create `AccessControl` dataclass (owner, teams, tags)
- [ ] Create `SourceType` enum (PDF, DOCX, TXT, URL, API, etc.)
- [ ] Implement document serialization/deserialization
- [ ] Write tests

**Deliverable:** Can create and serialize Document objects

---

## Checkpoint 3.2: Ingestion Step Interface
- [ ] Create `BaseIngestionStep(Step)` abstract class
- [ ] Define standard ingestion input schema (source config)
- [ ] Define standard ingestion output schema (List[Document])
- [ ] Implement `ingest(source_config) -> List[Document]` abstract method
- [ ] Add pre-processing hook support
- [ ] Write tests with mock ingestion step

**Deliverable:** Base class for all ingestion steps defined

---

## Checkpoint 3.3: File Ingestors
- [ ] Implement `TextFileIngestionStep` (TXT files)
- [ ] Implement `PDFIngestionStep` (using pypdf or pdfplumber)
- [ ] Implement `DocxIngestionStep` (using python-docx)
- [ ] Implement `MarkdownIngestionStep`
- [ ] Add file path / glob pattern support
- [ ] Add encoding detection
- [ ] Register all steps in registry
- [ ] Write tests with sample files

**Deliverable:** Can ingest PDF, DOCX, TXT, MD files

---

## Checkpoint 3.4: Web & API Ingestors
- [ ] Implement `URLIngestionStep` (single URL)
- [ ] Implement `WebCrawlerIngestionStep` (multi-page)
- [ ] Implement `APIIngestionStep` (REST API polling)
- [ ] Add rate limiting for web requests
- [ ] Add HTML-to-text conversion (BeautifulSoup/trafilatura)
- [ ] Add robots.txt respect option
- [ ] Write tests with mock server

**Deliverable:** Can ingest from URLs and APIs

---

## Checkpoint 3.5: Ingestion Pipeline Features
- [ ] Implement document deduplication (hash-based)
- [ ] Implement incremental ingestion (track last ingested)
- [ ] Implement re-ingestion policy (on_change, scheduled)
- [ ] Create `IngestionState` storage interface
- [ ] Implement file-based ingestion state store
- [ ] Write tests for incremental ingestion

**Deliverable:** Smart ingestion with dedup and incremental support

---

# PHASE 4: CHUNKING SYSTEM

## Checkpoint 4.1: Chunk Model
- [ ] Create `Chunk` dataclass:
  - `chunk_id: str`
  - `doc_id: str` (parent reference)
  - `content: str`
  - `index: int` (position in document)
  - `metadata: dict` (inherited + chunk-specific)
  - `token_count: int`
  - `parent_chunk_id: Optional[str]` (for hierarchical)
- [ ] Implement chunk serialization
- [ ] Write tests

**Deliverable:** Chunk model ready

---

## Checkpoint 4.2: Chunking Step Interface
- [ ] Create `BaseChunkingStep(Step)` abstract class
- [ ] Define input schema (List[Document])
- [ ] Define output schema (List[Chunk])
- [ ] Implement `chunk(documents) -> List[Chunk]` abstract method
- [ ] Add metadata inheritance logic
- [ ] Write tests

**Deliverable:** Base chunking interface defined

---

## Checkpoint 4.3: Basic Chunking Strategies
- [ ] Implement `FixedSizeChunkingStep` (char count)
- [ ] Implement `TokenAwareChunkingStep` (uses tokenizer)
- [ ] Add configurable overlap (fixed and percentage)
- [ ] Add configurable separator (newline, sentence, etc.)
- [ ] Register steps
- [ ] Write tests with various configs

**Deliverable:** Fixed and token-aware chunking working

---

## Checkpoint 4.4: Advanced Chunking Strategies
- [ ] Implement `SemanticChunkingStep` (embedding-based boundaries)
- [ ] Implement `LayoutAwareChunkingStep` (respects headings/sections)
- [ ] Implement `HierarchicalChunkingStep` (parent/child chunks)
- [ ] Add language detection for language-aware chunking
- [ ] Add section preservation option
- [ ] Write tests

**Deliverable:** All chunking strategies implemented

---

# PHASE 5: EMBEDDING SYSTEM

## Checkpoint 5.1: Embedding Step Interface
- [ ] Create `BaseEmbeddingStep(Step)` abstract class
- [ ] Define input schema (List[Chunk] or List[str])
- [ ] Define output schema (List[EmbeddedChunk])
- [ ] Create `EmbeddedChunk` dataclass (chunk + vector)
- [ ] Define `EmbeddingModel` protocol (abstract interface)
- [ ] Write tests

**Deliverable:** Embedding interface defined

---

## Checkpoint 5.2: Embedding Providers
- [ ] Implement `OpenAIEmbeddingModel` (text-embedding-3-small/large)
- [ ] Implement `LocalEmbeddingModel` (sentence-transformers)
- [ ] Implement `CohereEmbeddingModel`
- [ ] Add provider config (api_key, model_name, dimensions)
- [ ] Add batch embedding support
- [ ] Add rate limiting and retry
- [ ] Write tests (with mocks for API providers)

**Deliverable:** Multiple embedding providers working

---

## Checkpoint 5.3: Advanced Embedding Features
- [ ] Implement hybrid embeddings (dense + sparse/BM25)
- [ ] Implement embedding versioning (track model version)
- [ ] Implement late binding (embed at query time)
- [ ] Add embedding caching layer
- [ ] Write tests

**Deliverable:** Advanced embedding features complete

---

# PHASE 6: INDEXING SYSTEM

## Checkpoint 6.1: Index Interface
- [ ] Create `BaseIndex` abstract class:
  - `add(chunks: List[EmbeddedChunk])`
  - `search(query_vector, top_k, filters) -> List[SearchResult]`
  - `delete(chunk_ids: List[str])`
  - `get_stats() -> IndexStats`
- [ ] Create `SearchResult` dataclass (chunk, score, metadata)
- [ ] Create `IndexStats` dataclass (count, size, etc.)
- [ ] Write tests with mock index

**Deliverable:** Index interface defined

---

## Checkpoint 6.2: Vector Index Implementations
- [ ] Implement `InMemoryVectorIndex` (numpy/faiss)
- [ ] Implement `ChromaDBIndex` wrapper
- [ ] Implement `QdrantIndex` wrapper
- [ ] Implement `PineconeIndex` wrapper
- [ ] Add namespace/collection support
- [ ] Add metadata filtering support
- [ ] Write integration tests

**Deliverable:** Multiple vector stores supported

---

## Checkpoint 6.3: Hybrid & Advanced Indexing
- [ ] Implement `BM25Index` (sparse)
- [ ] Implement `HybridIndex` (combines dense + sparse)
- [ ] Implement index versioning (v1, v2, etc.)
- [ ] Implement hot/cold index separation
- [ ] Implement rebuild strategies (full, incremental)
- [ ] Write tests

**Deliverable:** Hybrid indexing working

---

## Checkpoint 6.4: Indexing Step
- [ ] Create `IndexingStep(Step)` class
- [ ] Wire up to index interface
- [ ] Add upsert logic (add or update)
- [ ] Add batch indexing
- [ ] Register step
- [ ] Write tests

**Deliverable:** Indexing as a pipeline step

---

# PHASE 7: RETRIEVAL SYSTEM

## Checkpoint 7.1: Retrieval Step Interface
- [ ] Create `BaseRetrievalStep(Step)` abstract class
- [ ] Define input schema (query: str, filters: dict)
- [ ] Define output schema (List[RetrievedChunk])
- [ ] Create `RetrievedChunk` dataclass (chunk + score + metadata)
- [ ] Write tests

**Deliverable:** Retrieval interface defined

---

## Checkpoint 7.2: Basic Retrieval Strategies
- [ ] Implement `TopKRetrievalStep` (simple top-k)
- [ ] Implement `HybridRetrievalStep` (dense + sparse fusion)
- [ ] Implement `MetadataFilteredRetrievalStep`
- [ ] Add score threshold filtering
- [ ] Add deduplication of results
- [ ] Register steps
- [ ] Write tests

**Deliverable:** Basic retrieval working

---

## Checkpoint 7.3: Advanced Retrieval Strategies
- [ ] Implement `MultiQueryRetrievalStep` (generate query variants)
- [ ] Implement `QueryRewritingRetrievalStep` (LLM rewrites query)
- [ ] Implement `MultiHopRetrievalStep` (iterative retrieval)
- [ ] Implement confidence-based depth retrieval
- [ ] Write tests

**Deliverable:** Advanced retrieval strategies complete

---

## Checkpoint 7.4: Context-Aware Retrieval
- [ ] Implement user-context-aware retrieval (uses RuntimeContext)
- [ ] Implement role/department-based filtering
- [ ] Implement time-aware retrieval (recency boost)
- [ ] Implement ACL-based filtering
- [ ] Write tests

**Deliverable:** Retrieval respects context and permissions

---

# PHASE 8: RERANKING SYSTEM

## Checkpoint 8.1: Reranking Step Interface
- [ ] Create `BaseRerankingStep(Step)` abstract class
- [ ] Define input schema (query, List[RetrievedChunk])
- [ ] Define output schema (List[RerankedChunk] with new scores)
- [ ] Create `RerankedChunk` dataclass
- [ ] Write tests

**Deliverable:** Reranking interface defined

---

## Checkpoint 8.2: Reranking Implementations
- [ ] Implement `CrossEncoderRerankingStep` (sentence-transformers)
- [ ] Implement `CohereRerankingStep` (Cohere rerank API)
- [ ] Implement `LLMRerankingStep` (use LLM to score relevance)
- [ ] Implement `RuleBasedRerankingStep` (custom scoring rules)
- [ ] Add score threshold and dynamic top-n
- [ ] Add cost-aware reranking (limit expensive rerankers)
- [ ] Register steps
- [ ] Write tests

**Deliverable:** Multiple reranking strategies available

---

# PHASE 9: PROMPT ASSEMBLY

## Checkpoint 9.1: Prompt Template System
- [ ] Create `PromptTemplate` class:
  - `template_id: str`
  - `version: str`
  - `template: str` (with {{placeholders}})
  - `variables: List[str]`
  - `output_format: OutputFormat` (text, json, etc.)
- [ ] Implement Jinja2-based rendering
- [ ] Implement template validation (all vars provided)
- [ ] Implement template versioning storage
- [ ] Write tests

**Deliverable:** Template system working

---

## Checkpoint 9.2: Prompt Assembly Step
- [ ] Create `PromptAssemblyStep(Step)` class
- [ ] Define input schema (query, retrieved_chunks, template_id)
- [ ] Define output schema (assembled_prompt: str)
- [ ] Implement context ordering strategies (relevance, recency, etc.)
- [ ] Implement token budgeting (truncate to fit)
- [ ] Implement context compression (summarize if too long)
- [ ] Add citation injection (add source markers)
- [ ] Register step
- [ ] Write tests

**Deliverable:** Intelligent prompt assembly working

---

## Checkpoint 9.3: Multi-Role Prompts
- [ ] Support system/user/assistant message format
- [ ] Create `ChatPromptTemplate` for chat models
- [ ] Support structured output prompts (JSON mode)
- [ ] Write tests

**Deliverable:** Chat and structured prompts supported

---

# PHASE 10: LLM EXECUTION

## Checkpoint 10.1: LLM Provider Interface
- [ ] Create `BaseLLMProvider` abstract class:
  - `generate(prompt, config) -> LLMResponse`
  - `generate_stream(prompt, config) -> AsyncIterator[str]`
- [ ] Create `LLMConfig` dataclass (temperature, max_tokens, etc.)
- [ ] Create `LLMResponse` dataclass (text, usage, latency)
- [ ] Write tests

**Deliverable:** LLM interface defined

---

## Checkpoint 10.2: LLM Provider Implementations
- [ ] Implement `OpenAIProvider` (GPT-4, GPT-3.5)
- [ ] Implement `AnthropicProvider` (Claude)
- [ ] Implement `GoogleProvider` (Gemini)
- [ ] Implement `LocalLLMProvider` (Ollama/vLLM)
- [ ] Add API key management
- [ ] Add rate limiting
- [ ] Add retry logic
- [ ] Write tests

**Deliverable:** Multiple LLM providers supported

---

## Checkpoint 10.3: LLM Execution Step
- [ ] Create `LLMExecutionStep(Step)` class
- [ ] Define input schema (prompt, config)
- [ ] Define output schema (response, usage, citations)
- [ ] Implement model fallback chains (try GPT-4, fallback to GPT-3.5)
- [ ] Implement cost caps (stop if budget exceeded)
- [ ] Implement streaming support
- [ ] Register step
- [ ] Write tests

**Deliverable:** LLM execution as pipeline step

---

## Checkpoint 10.4: Advanced LLM Features
- [ ] Implement tool/function calling support
- [ ] Implement multi-call orchestration (agent-like)
- [ ] Implement response caching
- [ ] Write tests

**Deliverable:** Advanced LLM features available

---

# PHASE 11: POST-PROCESSING

## Checkpoint 11.1: Post-Processing Step Interface
- [ ] Create `BasePostProcessingStep(Step)` abstract class
- [ ] Define input schema (llm_response, context)
- [ ] Define output schema (processed_response)
- [ ] Write tests

**Deliverable:** Post-processing interface defined

---

## Checkpoint 11.2: Output Processing
- [ ] Implement `JSONValidationStep` (validate JSON output)
- [ ] Implement `SchemaEnforcementStep` (coerce to schema)
- [ ] Implement `CitationFormattingStep` (format source citations)
- [ ] Implement `ConfidenceScoringStep` (add confidence score)
- [ ] Register steps
- [ ] Write tests

**Deliverable:** Output processing steps complete

---

## Checkpoint 11.3: Safety & Compliance
- [ ] Implement `PIIMaskingStep` (detect and mask PII)
- [ ] Implement `RedactionStep` (custom redaction rules)
- [ ] Implement `GroundingCheckStep` (verify answer in sources)
- [ ] Add configurable rules
- [ ] Write tests

**Deliverable:** Safety post-processing complete

---

# PHASE 12: STORAGE LAYER

## Checkpoint 12.1: Storage Interface
- [ ] Create `BaseStorage` abstract class:
  - `save_document(doc: Document)`
  - `get_document(doc_id: str) -> Document`
  - `save_chunks(chunks: List[Chunk])`
  - `get_chunks(doc_id: str) -> List[Chunk]`
  - `save_pipeline(spec: PipelineSpec)`
  - `get_pipeline(name, version) -> PipelineSpec`
- [ ] Write tests

**Deliverable:** Storage interface defined

---

## Checkpoint 12.2: Storage Implementations
- [ ] Implement `FileSystemStorage` (JSON files)
- [ ] Implement `SQLiteStorage`
- [ ] Implement `PostgresStorage` (optional)
- [ ] Add connection pooling
- [ ] Write tests

**Deliverable:** Multiple storage backends

---

# PHASE 13: TRACING & OBSERVABILITY

## Checkpoint 13.1: Trace Model
- [ ] Create `Trace` dataclass:
  - `trace_id: str`
  - `pipeline_name: str`
  - `pipeline_version: str`
  - `start_time: datetime`
  - `end_time: datetime`
  - `steps: List[StepTrace]`
  - `success: bool`
  - `error: Optional[str]`
- [ ] Create `StepTrace` dataclass:
  - `step_id: str`
  - `input: Any`
  - `output: Any`
  - `latency_ms: float`
  - `token_usage: TokenUsage`
  - `error: Optional[str]`
- [ ] Create `TokenUsage` dataclass
- [ ] Write tests

**Deliverable:** Trace model complete

---

## Checkpoint 13.2: Trace Collection
- [ ] Instrument PipelineExecutor to emit traces
- [ ] Create `TraceCollector` class
- [ ] Implement in-memory trace storage
- [ ] Implement file-based trace storage
- [ ] Add trace export (JSON)
- [ ] Write tests

**Deliverable:** Automatic trace collection working

---

## Checkpoint 13.3: Metrics
- [ ] Create `Metrics` class:
  - Total cost
  - Total latency
  - Per-step latency
  - Token usage breakdown
  - Retrieval count
- [ ] Implement metrics aggregation
- [ ] Add metrics export (Prometheus format optional)
- [ ] Write tests

**Deliverable:** Metrics collection working

---

# PHASE 14: EVALUATION SYSTEM

## Checkpoint 14.1: Evaluation Dataset
- [ ] Create `EvalDataset` class:
  - `questions: List[str]`
  - `expected_answers: List[str]`
  - `expected_sources: List[List[str]]`
- [ ] Implement dataset loading (JSON/CSV)
- [ ] Write tests

**Deliverable:** Eval dataset support

---

## Checkpoint 14.2: Evaluation Metrics
- [ ] Implement retrieval recall metric
- [ ] Implement retrieval precision metric
- [ ] Implement answer faithfulness metric (LLM-based)
- [ ] Implement answer relevance metric
- [ ] Implement hallucination detection metric
- [ ] Write tests

**Deliverable:** Core eval metrics implemented

---

## Checkpoint 14.3: Evaluation Harness
- [ ] Create `EvaluationHarness` class
- [ ] Implement `run_eval(pipeline, dataset) -> EvalResults`
- [ ] Implement A/B comparison (two pipelines)
- [ ] Implement regression detection
- [ ] Generate eval reports
- [ ] Write tests

**Deliverable:** Full evaluation system working

---

# PHASE 15: SDK & API

## Checkpoint 15.1: Python SDK
- [ ] Create `RAGClient` class:
  - `load_pipeline(name, version)`
  - `run(query, overrides)`
  - `run_batch(queries)`
  - `get_trace(trace_id)`
- [ ] Add type hints throughout
- [ ] Write SDK documentation
- [ ] Write tests

**Deliverable:** Python SDK complete

---

## Checkpoint 15.2: REST API
- [ ] Create FastAPI application
- [ ] Implement `POST /run` endpoint
- [ ] Implement `GET /pipelines` endpoint
- [ ] Implement `GET /pipelines/{name}/versions` endpoint
- [ ] Implement `GET /traces/{trace_id}` endpoint
- [ ] Implement `GET /metrics` endpoint
- [ ] Add authentication middleware
- [ ] Write API tests

**Deliverable:** REST API complete

---

## Checkpoint 15.3: CLI
- [ ] Create CLI using Click/Typer
- [ ] Implement `rag run <pipeline> <query>`
- [ ] Implement `rag validate <pipeline.yaml>`
- [ ] Implement `rag list-pipelines`
- [ ] Implement `rag eval <pipeline> <dataset>`
- [ ] Write CLI tests

**Deliverable:** CLI complete

---

# PHASE 16: PLUGIN SYSTEM

## Checkpoint 16.1: Plugin Interface
- [ ] Define plugin entry point specification
- [ ] Create `BasePlugin` abstract class
- [ ] Implement plugin discovery (entry_points)
- [ ] Implement plugin loading
- [ ] Implement plugin versioning
- [ ] Write tests

**Deliverable:** Plugin system foundation

---

## Checkpoint 16.2: Plugin Management
- [ ] Implement `PluginManager` class
- [ ] Add plugin enable/disable
- [ ] Add plugin configuration
- [ ] Add plugin dependency resolution
- [ ] Create sample plugin
- [ ] Write documentation

**Deliverable:** Full plugin system working

---

# PHASE 17: SECURITY & OPS

## Checkpoint 17.1: Security Features
- [ ] Implement secrets manager integration (env vars, vault)
- [ ] Implement audit logging
- [ ] Implement RBAC at document level
- [ ] Implement data isolation (tenant separation)
- [ ] Write tests

**Deliverable:** Security features complete

---

## Checkpoint 17.2: Ops Features
- [ ] Implement rate limiting (per user/pipeline)
- [ ] Implement cost monitoring and alerts
- [ ] Implement health checks
- [ ] Add alerting hooks (webhook)
- [ ] Write tests

**Deliverable:** Ops features complete

---

# PHASE 18: UI (OPTIONAL / FUTURE)

## Checkpoint 18.1: Pipeline Builder UI
- [ ] Set up React/Next.js project
- [ ] Create pipeline canvas component
- [ ] Implement drag-and-drop steps
- [ ] Implement step configuration panels
- [ ] Implement schema preview
- [ ] Implement compatibility warnings

**Deliverable:** Visual pipeline builder

---

## Checkpoint 18.2: Debug & Trace UI
- [ ] Create trace viewer component
- [ ] Show step-by-step execution
- [ ] Show intermediate inputs/outputs
- [ ] Show token and cost breakdown
- [ ] Implement test query runner

**Deliverable:** Debug UI complete

---

## Checkpoint 18.3: Version Management UI
- [ ] Show pipeline version history
- [ ] Implement version diff view
- [ ] Implement rollback functionality
- [ ] Implement version promotion

**Deliverable:** Version management UI complete

---

# SUMMARY: BUILD ORDER

```
Phase 1:  Foundation (Steps, Specs, Registry, Validation)
Phase 2:  Execution Engine (Context, Executor, Errors, Async)
Phase 3:  Ingestion (Documents, File/Web loaders)
Phase 4:  Chunking (All strategies)
Phase 5:  Embedding (Providers, Hybrid)
Phase 6:  Indexing (Vector stores, Hybrid)
Phase 7:  Retrieval (Basic + Advanced)
Phase 8:  Reranking
Phase 9:  Prompt Assembly
Phase 10: LLM Execution
Phase 11: Post-Processing
Phase 12: Storage Layer
Phase 13: Tracing & Observability
Phase 14: Evaluation System
Phase 15: SDK & API
Phase 16: Plugin System
Phase 17: Security & Ops
Phase 18: UI (Future)
```

---

# TESTING STRATEGY

Each checkpoint should have:
1. **Unit tests** for individual components
2. **Integration tests** for step interactions
3. **End-to-end test** after each phase

Run tests before moving to next checkpoint.

---

# NOTES

- Each checkpoint is designed to be completable independently
- Always write tests before or alongside implementation
- Document public APIs as you build
- Keep interfaces stable; implementation can change
- Use dependency injection for testability

---

**Total Checkpoints: 58**
**Total Phases: 18**

Ready to start building. Begin with Checkpoint 1.1.
