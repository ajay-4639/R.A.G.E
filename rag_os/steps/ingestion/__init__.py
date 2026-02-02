"""Ingestion steps for RAG OS."""

from rag_os.steps.ingestion.base import BaseIngestionStep, IngestionConfig
from rag_os.steps.ingestion.file_ingestors import (
    TextFileIngestionStep,
    MarkdownIngestionStep,
    JSONIngestionStep,
    CSVIngestionStep,
)
from rag_os.steps.ingestion.pdf_ingestor import PDFIngestionStep
from rag_os.steps.ingestion.web_ingestors import (
    URLIngestionStep,
    WebCrawlerIngestionStep,
    APIIngestionStep,
)

__all__ = [
    "BaseIngestionStep",
    "IngestionConfig",
    # File ingestors
    "TextFileIngestionStep",
    "MarkdownIngestionStep",
    "JSONIngestionStep",
    "CSVIngestionStep",
    "PDFIngestionStep",
    # Web ingestors
    "URLIngestionStep",
    "WebCrawlerIngestionStep",
    "APIIngestionStep",
]
