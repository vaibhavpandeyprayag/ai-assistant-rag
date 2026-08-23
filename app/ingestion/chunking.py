"""Configurable text chunking.

Strategies:

- ``recursive`` (default): splits preferring paragraph, then sentence, then
  word boundaries via ``langchain-text-splitters``.
- ``fixed``: a sliding character window with ``size - overlap`` step.

Chunk size/overlap/strategy always come from configuration (``Settings``) and
are never hard-coded here. The documented defaults (1000 chars / 200 overlap)
fit the ``all-MiniLM-L6-v2`` embedding window (~256 tokens) while ~20 percent
overlap keeps sentences that straddle boundaries intact.

Sections are cleaned and chunked independently so page attribution remains
exact: every chunk of a section inherits that section's page number.
"""

from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.cleaning import clean_text
from app.ingestion.models import Chunk, LoadedDocument

ChunkingStrategy = Literal["recursive", "fixed"]


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` into overlapping fixed-size windows."""
    pieces: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        pieces.append(text[start : start + chunk_size])
        # Stop once this window already reached the end of the text.
        if start + chunk_size >= len(text):
            break
        start += step
    return pieces


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` along natural boundaries using LangChain's splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    strategy: ChunkingStrategy = "recursive",
) -> list[str]:
    """Split raw ``text`` into chunk strings.

    Raises:
        ValueError: If sizes are invalid or overlap reaches chunk size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    # Defense in depth: Settings validates this too, but chunk_text is public.
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    cleaned = clean_text(text)
    if not cleaned:
        return []

    if strategy == "fixed":
        return _split_fixed(cleaned, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return _split_recursive(cleaned, chunk_size, chunk_overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def chunk_document(
    document: LoadedDocument,
    document_id: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: ChunkingStrategy = "recursive",
) -> list[Chunk]:
    """Turn a loaded document into retrieval-ready chunks.

    Chunks are numbered sequentially across the whole document and identified
    as ``"{document_id}::{index}"``. Sections whose text disappears during
    cleaning contribute no chunks.
    """
    chunks: list[Chunk] = []
    index = 0
    for section in document.sections:
        pieces = chunk_text(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
        )
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}::{index}",
                    document_id=document_id,
                    chunk_index=index,
                    text=piece,
                    page_number=section.page_number,
                    filename=document.filename,
                    source=document.source,
                )
            )
            index += 1
    return chunks
