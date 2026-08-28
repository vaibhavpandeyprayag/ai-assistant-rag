"""Shared FastAPI dependency providers.

Providers are cached so expensive objects (settings, and the embedder, vector
store, LLM client, retriever and RAG pipeline) are constructed once per
process. Tests replace them via ``app.dependency_overrides``.
"""

from functools import lru_cache

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import create_embedding_provider
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever, create_retriever
from app.services.chat_service import ChatService
from app.services.ingestion_service import (
    IngestionService,
    create_ingestion_service,
)
from app.vectordb.base import VectorStoreRepository
from app.vectordb.chroma import ChromaVectorStoreRepository


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""
    return Settings()


@lru_cache
def get_vector_store() -> VectorStoreRepository:
    """Return a persistent local vector store (Chroma) singleton."""
    settings = get_settings()
    return ChromaVectorStoreRepository(settings.chroma_persist_directory)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider."""
    return create_embedding_provider(get_settings())


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider (Hugging Face Inference API)."""
    return create_llm_provider(get_settings())


@lru_cache
def get_ingestion_service() -> IngestionService:
    """Return an ingestion service wired to the configured embedder/store."""
    return create_ingestion_service(
        get_settings(), get_embedding_provider(), get_vector_store()
    )


@lru_cache
def get_retriever() -> Retriever:
    """Return a query-time retriever wired to the configured embedder/store."""
    return create_retriever(get_settings(), get_embedding_provider(), get_vector_store())


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    """Return the RAG pipeline wired to the configured retriever and LLM."""
    return RAGPipeline(
        get_retriever(),
        get_llm_provider(),
        top_k=get_settings().top_k,
    )


@lru_cache
def get_chat_service() -> ChatService:
    """Return a chat service backed by the configured RAG pipeline."""
    return ChatService(get_rag_pipeline())
