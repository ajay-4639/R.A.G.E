"""Chunking strategy implementations."""

import re
from typing import Any

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.document import Document
from rag_os.models.chunk import Chunk
from rag_os.steps.chunking.base import BaseChunkingStep


@register_step(
    name="FixedSizeChunkingStep",
    step_type=StepType.CHUNKING,
    description="Chunks text by fixed character count",
    version="1.0.0",
)
class FixedSizeChunkingStep(BaseChunkingStep):
    """
    Chunks documents by fixed character count.

    Simple chunking strategy that splits text into chunks of
    approximately equal size with configurable overlap.
    """

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks."""
        config = self.get_config()
        text = document.content
        chunks: list[Chunk] = []

        chunk_size = config.chunk_size
        overlap = config.chunk_overlap

        # Validate overlap
        if overlap >= chunk_size:
            overlap = chunk_size // 4

        start = 0
        index = 0

        while start < len(text):
            # Calculate end position
            end = start + chunk_size

            # Don't go past the end
            if end > len(text):
                end = len(text)

            # Extract chunk content
            content = text[start:end]

            # Skip if too small (except for last chunk)
            if len(content) < config.min_chunk_size and index > 0:
                break

            chunk = self._create_chunk(
                content=content,
                doc_id=document.doc_id,
                index=index,
                start_char=start,
                metadata=document.metadata.copy() if config.include_metadata else {},
            )
            chunks.append(chunk)

            # Move start position (accounting for overlap)
            start = end - overlap
            index += 1

            # Prevent infinite loop
            if start >= len(text) - overlap:
                break

        return chunks


@register_step(
    name="SentenceChunkingStep",
    step_type=StepType.CHUNKING,
    description="Chunks text by sentence boundaries",
    version="1.0.0",
)
class SentenceChunkingStep(BaseChunkingStep):
    """
    Chunks documents by sentence boundaries.

    Splits text into sentences first, then groups sentences
    to reach the target chunk size while respecting boundaries.
    """

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into sentence-based chunks."""
        config = self.get_config()
        text = document.content
        chunks: list[Chunk] = []

        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return chunks

        current_chunk: list[str] = []
        current_size = 0
        chunk_start = 0
        index = 0
        char_pos = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would exceed chunk_size
            if current_size + sentence_len > config.chunk_size and current_chunk:
                # Save current chunk
                content = " ".join(current_chunk)
                chunk = self._create_chunk(
                    content=content,
                    doc_id=document.doc_id,
                    index=index,
                    start_char=chunk_start,
                    metadata=document.metadata.copy() if config.include_metadata else {},
                )
                chunks.append(chunk)
                index += 1

                # Handle overlap by keeping some sentences
                overlap_sentences = self._calculate_overlap_sentences(
                    current_chunk, config.chunk_overlap
                )
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
                chunk_start = char_pos - current_size

            current_chunk.append(sentence)
            current_size += sentence_len
            char_pos += sentence_len + 1  # +1 for space/separator

        # Don't forget the last chunk
        if current_chunk:
            content = " ".join(current_chunk)
            if len(content) >= config.min_chunk_size or not chunks:
                chunk = self._create_chunk(
                    content=content,
                    doc_id=document.doc_id,
                    index=index,
                    start_char=chunk_start,
                    metadata=document.metadata.copy() if config.include_metadata else {},
                )
                chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting pattern
        # Handles common cases: periods, question marks, exclamation marks
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_overlap_sentences(
        self, sentences: list[str], target_overlap: int
    ) -> list[str]:
        """Get sentences to keep for overlap."""
        if not sentences or target_overlap <= 0:
            return []

        overlap_sentences: list[str] = []
        overlap_size = 0

        # Work backwards through sentences
        for sentence in reversed(sentences):
            if overlap_size + len(sentence) > target_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_size += len(sentence)

        return overlap_sentences


@register_step(
    name="TokenAwareChunkingStep",
    step_type=StepType.CHUNKING,
    description="Chunks text by token count (requires tokenizer)",
    version="1.0.0",
)
class TokenAwareChunkingStep(BaseChunkingStep):
    """
    Chunks documents by token count.

    Uses a tokenizer to count tokens and splits text to stay
    within token limits. Useful for LLM context windows.
    """

    def __init__(
        self,
        step_id: str,
        step_type: StepType = StepType.CHUNKING,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(step_id, step_type, config)
        self._tokenizer = None

    def _get_tokenizer(self):
        """Get or create tokenizer."""
        if self._tokenizer is None:
            tokenizer_name = self.config.get("tokenizer", "cl100k_base")
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(tokenizer_name)
            except ImportError:
                # Fallback to simple word-based tokenization
                self._tokenizer = None
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Simple fallback: approximate 4 characters per token
            return len(text) // 4

    def _encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return tokenizer.encode(text)
        else:
            # Fallback: use character positions
            return list(range(0, len(text), 4))

    def _decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return tokenizer.decode(tokens)
        else:
            # Fallback not applicable
            return ""

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into token-based chunks."""
        config = self.get_config()
        text = document.content
        chunks: list[Chunk] = []

        tokenizer = self._get_tokenizer()

        if tokenizer:
            # Token-based chunking
            tokens = tokenizer.encode(text)
            total_tokens = len(tokens)

            chunk_tokens = config.chunk_size  # chunk_size is in tokens
            overlap_tokens = config.chunk_overlap

            start = 0
            index = 0

            while start < total_tokens:
                end = min(start + chunk_tokens, total_tokens)

                chunk_token_slice = tokens[start:end]
                content = tokenizer.decode(chunk_token_slice)

                chunk = self._create_chunk(
                    content=content,
                    doc_id=document.doc_id,
                    index=index,
                    start_char=0,  # Token-based doesn't track char positions easily
                    metadata={
                        **(document.metadata.copy() if config.include_metadata else {}),
                        "token_count": len(chunk_token_slice),
                    },
                )
                chunk.token_count = len(chunk_token_slice)
                chunks.append(chunk)

                start = end - overlap_tokens
                index += 1

                if start >= total_tokens - overlap_tokens:
                    break
        else:
            # Fallback to character-based with token estimation
            chars_per_token = 4
            chunk_chars = config.chunk_size * chars_per_token
            overlap_chars = config.chunk_overlap * chars_per_token

            fallback_step = FixedSizeChunkingStep(
                step_id=self.step_id + "_fallback",
                config={
                    "chunk_size": chunk_chars,
                    "chunk_overlap": overlap_chars,
                    **self.config,
                },
            )
            chunks = fallback_step.chunk_document(document)

            # Estimate token counts
            for chunk in chunks:
                chunk.token_count = len(chunk.content) // chars_per_token

        return chunks


@register_step(
    name="RecursiveChunkingStep",
    step_type=StepType.CHUNKING,
    description="Recursively chunks text using multiple separators",
    version="1.0.0",
)
class RecursiveChunkingStep(BaseChunkingStep):
    """
    Recursively chunks documents using a hierarchy of separators.

    Tries to split by larger semantic units first (paragraphs),
    then falls back to smaller units (sentences, words) if needed.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        ", ",    # Clauses
        " ",     # Words
        "",      # Characters
    ]

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document using recursive strategy."""
        config = self.get_config()
        text = document.content

        # Get separators from config or use defaults
        separators = self.config.get("separators", self.DEFAULT_SEPARATORS)

        # Recursively split
        text_chunks = self._recursive_split(
            text=text,
            separators=separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Create Chunk objects
        chunks: list[Chunk] = []
        char_pos = 0

        for i, content in enumerate(text_chunks):
            if len(content) < config.min_chunk_size and i > 0:
                continue

            chunk = self._create_chunk(
                content=content,
                doc_id=document.doc_id,
                index=i,
                start_char=char_pos,
                metadata=document.metadata.copy() if config.include_metadata else {},
            )
            chunks.append(chunk)

            # Approximate char position (overlap makes this inexact)
            char_pos += len(content) - config.chunk_overlap

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            # Base case: just return the text
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator = split by character
            splits = list(text)

        # Merge splits into chunks of appropriate size
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for split in splits:
            split_size = len(split) + len(separator)

            if current_size + split_size > chunk_size and current_chunk:
                # Current chunk is full, save it
                chunk_text = separator.join(current_chunk)

                # If chunk is still too big, recursively split it
                if len(chunk_text) > chunk_size and remaining_separators:
                    sub_chunks = self._recursive_split(
                        chunk_text, remaining_separators, chunk_size, chunk_overlap
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_parts = self._get_overlap_parts(
                    current_chunk, separator, chunk_overlap
                )
                current_chunk = overlap_parts
                current_size = sum(len(p) + len(separator) for p in current_chunk)

            current_chunk.append(split)
            current_size += split_size

        # Handle remaining content
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(
                    chunk_text, remaining_separators, chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return chunks

    def _get_overlap_parts(
        self, parts: list[str], separator: str, target_overlap: int
    ) -> list[str]:
        """Get parts to keep for overlap."""
        if not parts or target_overlap <= 0:
            return []

        overlap_parts: list[str] = []
        overlap_size = 0

        for part in reversed(parts):
            part_size = len(part) + len(separator)
            if overlap_size + part_size > target_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_size += part_size

        return overlap_parts
