"""Semantic search endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_retriever
from app.retrieval.retriever import Retriever
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> SearchResponse:
    """Embed the query and return the top-K semantically similar chunks."""
    results = retriever.retrieve(
        request.query,
        top_k=request.top_k,
        filter=request.filter,
    )
    return SearchResponse(
        results=[
            SearchResultItem(text=item.text, score=item.score, metadata=item.metadata)
            for item in results
        ]
    )
