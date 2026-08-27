"""RAG prompt templates.

These strings are kept separate from application logic so they can be edited
and evaluated independently (per the project's prompting guidance). The system
prompt encodes the grounding rules the pipeline relies on.
"""

from app.llm.base import ChatMessage
from app.rag.context_builder import BuiltContext

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "retrieved context provided below.\n"
    "Rules:\n"
    "1. Answer using the context only. Do not use outside knowledge.\n"
    "2. After each claim that comes from the context, cite its source block(s) "
    "like [1] or [1][2].\n"
    "3. If the context does not contain enough information to answer the "
    "question, say so clearly instead of guessing.\n"
    "4. If the context is empty or irrelevant, state that you cannot answer "
    "based on the provided documents.\n"
    "5. Never invent facts, URLs, or citations that are not present in the "
    "context.\n"
)


def build_rag_messages(
    *,
    query: str,
    context: BuiltContext,
) -> list[ChatMessage]:
    """Build the chat messages carrying system rules, context, and query."""
    user_content = f"Context:\n{context.text}\n\nQuestion: {query}"
    return [
        ChatMessage(role="system", content=RAG_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
