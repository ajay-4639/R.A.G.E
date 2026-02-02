"""Core type definitions for RAG OS."""

from enum import Enum
from typing import Any


class StepType(str, Enum):
    """Enumeration of all step types in a RAG pipeline."""

    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    PROMPT_ASSEMBLY = "prompt_assembly"
    LLM_EXECUTION = "llm_execution"
    POST_PROCESSING = "post_processing"

    def __str__(self) -> str:
        return self.value


# Type aliases for clarity
SchemaType = dict[str, Any]
ConfigType = dict[str, Any]
MetadataType = dict[str, Any]
