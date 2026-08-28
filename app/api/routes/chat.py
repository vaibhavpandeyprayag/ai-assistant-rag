"""Grounded chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Answer a question using retrieved context and supporting sources."""
    return service.answer(request.query, top_k=request.top_k)
