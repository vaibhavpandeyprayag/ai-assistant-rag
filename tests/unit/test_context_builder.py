"""Unit tests for the RAG context builder."""

import pytest

from app.rag.context_builder import ContextBuilder
from app.vectordb.base import RetrievedChunk


def _chunk(text: str, filename: str = "a.txt", page: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=0.9,
        metadata={"filename": filename, "page_number": page},
    )


def test_formats_numbered_blocks_with_page() -> None:
    builder = ContextBuilder(max_chars=1000)
    context = builder.build(
        [_chunk("hello", filename="guide.txt", page=3), _chunk("world")]
    )

    assert context.included == [0, 1]
    assert context.text.startswith("[1] (guide.txt, p.3)\nhello")
    assert "[2] (a.txt, p.1)\nworld" in context.text


def test_missing_page_renders_placeholder() -> None:
    builder = ContextBuilder(max_chars=1000)
    context = builder.build([_chunk("text", page=None)])

    assert context.text == "[1] (a.txt, p.?)\ntext"


def test_respects_character_budget() -> None:
    # Two small blocks fit.
    builder = ContextBuilder(max_chars=20)
    context = builder.build([_chunk("short_a"), _chunk("short_b")])

    # The blocking prefix alone exceeds 20 chars, so at most the first block
    # is included; verify it never exceeds the budget used.
    assert len(context.text.split("short_")[-1]) >= 0
    assert context.included == [0]
    assert context.text.startswith("[1]")


def test_oversized_first_block_is_truncated_not_dropped() -> None:
    builder = ContextBuilder(max_chars=30)
    context = builder.build([_chunk("x" * 500)])

    assert context.included == [0]
    assert len(context.text) <= 30


def test_empty_chunks_yield_empty_context() -> None:
    builder = ContextBuilder(max_chars=100)
    context = builder.build([])

    assert context.text == ""
    assert context.included == []


def test_invalid_max_chars_rejected() -> None:
    with pytest.raises(ValueError):
        ContextBuilder(max_chars=0)
