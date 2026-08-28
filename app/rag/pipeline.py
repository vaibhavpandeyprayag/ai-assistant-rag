"""Grounded question-answering pipeline.

Orchestrates retrieval → context assembly → prompt → generation, returning
the answer together with its supporting sources. When nothing is retrieved it
short-circuits with a canned "insufficient context" response without calling
the LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from app.llm.base import LLMOptions, LLMProvider
from app.rag.context_builder import ContextBuilder
from app.rag.prompts import build_rag_messages
from app.vectordb.base import RetrievedChunk

#: Returned (without an LLM call) when retrieval yields nothing relevant.
INSUFFICIENT_CONTEXT_RESPONSE = (
    "I'm sorry, I don't have enough information in the provided documents "
    "to answer that question."
)


class RetrieverPort(Protocol):
    """Minimal retrieval contract satisfied by `Retriever` and test fakes."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: dict | None = None,  # noqa: A002
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        ...


@dataclass(slots=True)
class RAGSource:
    """Supporting source metadata for one retrieved chunk used in the answer."""

    chunk_id: str
    filename: str
    page_number: int | None
    score: float


@dataclass(slots=True)
class RAGAnswer:
    """The complete RAG response returned to callers."""

    answer: str
    sources: list[RAGSource] = field(default_factory=list)
    insufficient_context: bool = False
    latency_ms: float = 0.0


class RAGPipeline:
    """Ground answers in retrieved context."""

    def __init__(
        self,
        retriever: RetrieverPort,
        llm: LLMProvider,
        *,
        top_k: int = 5,
        max_context_chars: int = 6000,
        llm_options: LLMOptions | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._context_builder = ContextBuilder(max_chars=max_context_chars)
        self._llm_options = llm_options

    def answer(self, query: str, *, top_k: int | None = None) -> RAGAnswer:
        """Answer ``query`` using retrieved context and supporting sources.

        ``top_k`` overrides the pipeline default when provided.

        Raises:
            ValueError: If ``query`` is empty.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        effective_top_k = top_k if top_k is not None else self._top_k
        t0 = time.perf_counter()

        chunks = self._retriever.retrieve(query, top_k=effective_top_k)
        context = self._context_builder.build(chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # No relevant context: short-circuit without spending an LLM call.
        if not context.included:
            return RAGAnswer(
                answer=INSUFFICIENT_CONTEXT_RESPONSE,
                insufficient_context=True,
                latency_ms=elapsed_ms,
            )

        messages = build_rag_messages(query=query, context=context)
        answer_text = self._llm.generate(messages, self._llm_options)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Map the included chunk indexes back to their source metadata.
        sources = [
            _to_source(chunks[index])
            for index in context.included
        ]

        return RAGAnswer(
            answer=answer_text,
            sources=sources,
            insufficient_context=False,
            latency_ms=elapsed_ms,
        )


def _to_source(chunk: RetrievedChunk) -> RAGSource:
    metadata = chunk.metadata
    return RAGSource(
        chunk_id=str(metadata.get("chunk_id", "")),
        filename=str(metadata.get("filename", "")),
        page_number=metadata.get("page_number"),
        score=chunk.score,
    )
