"""Unit tests for the RAG prompt templates."""

from app.rag.context_builder import ContextBuilder
from app.rag.prompts import RAG_SYSTEM_PROMPT, build_rag_messages
from app.vectordb.base import RetrievedChunk


def _context() -> ContextBuilder:
    builder = ContextBuilder(max_chars=1000)
    chunk = RetrievedChunk(
        text="The sky is blue on clear days.",
        score=0.9,
        metadata={"filename": "facts.txt", "page_number": 2},
    )
    built = builder.build([chunk])
    assert built.included == [0]
    return built


def test_system_prompt_contains_grounding_rules() -> None:
    assert "using only the" in RAG_SYSTEM_PROMPT
    assert "cite its source block" in RAG_SYSTEM_PROMPT
    assert "does not contain enough information" in RAG_SYSTEM_PROMPT
    assert "Never invent facts" in RAG_SYSTEM_PROMPT


def test_build_messages_embed_context_and_query() -> None:
    messages = build_rag_messages(query="What color is the sky?", context=_context())

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "What color is the sky?" in messages[1].content
    assert "[1] (facts.txt, p.2)" in messages[1].content
    assert "The sky is blue" in messages[1].content
