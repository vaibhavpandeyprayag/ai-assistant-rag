"""Thin chat orchestration: RAG pipeline result → API response schema."""

from __future__ import annotations

from app.rag.pipeline import RAGPipeline
from app.schemas.chat import ChatResponse, ChatSource


class ChatService:
    """Translates a user question into a grounded, sourced answer.

    Keeps domain types (:class:`RAGAnswer`, :class:`RAGSource`) out of the
    HTTP layer so the pipeline stays independent of FastAPI/Pydantic.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def answer(self, query: str, *, top_k: int | None = None) -> ChatResponse:
        """Answer ``query`` and return the API-ready response."""
        result = self._pipeline.answer(query, top_k=top_k)
        return ChatResponse(
            answer=result.answer,
            sources=[
                ChatSource(
                    chunk_id=source.chunk_id,
                    filename=source.filename,
                    page_number=source.page_number,
                    score=source.score,
                )
                for source in result.sources
            ],
            insufficient_context=result.insufficient_context,
            latency_ms=result.latency_ms,
        )
