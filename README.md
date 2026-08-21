# AI Assistant with RAG

## Overview

Build a modular AI Assistant that uses Retrieval-Augmented Generation (RAG) to process user-provided documents and answer questions using relevant retrieved context.

The system will use:

- Python
- FastAPI
- LangChain
- ChromaDB
- Local LLMs
- Hugging Face models/APIs
- Pydantic
- pytest

## Core Features

### Document Processing

Support initial document formats:

- PDF
- DOCX
- TXT
- Markdown

Pipeline:

```text
Document → Load → Extract → Clean → Chunk → Embed → ChromaDB
```

Preserve useful metadata such as:

- document ID
- filename
- source
- chunk ID
- page number where available

### RAG Question Answering

```text
User Query
    ↓
Query Embedding
    ↓
ChromaDB Similarity Search
    ↓
Top-K Relevant Chunks
    ↓
Context Construction
    ↓
RAG Prompt
    ↓
LLM
    ↓
Answer + Sources
```

The LLM should answer using retrieved context, avoid unsupported claims, and indicate when relevant context is unavailable.

### LLM Support

Support both:

- Local LLMs
- Hugging Face models/APIs

LLM selection should be configuration-driven so the RAG pipeline does not depend on one provider.

### REST API

Initial endpoints:

```text
GET  /health
POST /documents/upload
POST /documents/ingest
POST /search
POST /chat
```

The exact API schemas should be finalized during planning.

## Architecture Requirements

Keep these concerns separate:

```text
FastAPI
   ↓
Application Services
   ↓
RAG Pipeline
   ├── Retriever
   ├── Context Builder
   ├── Prompt
   └── LLM
        ↑
   ChromaDB
        ↑
   Embeddings
        ↑
   Document Ingestion
```

FastAPI routes should remain thin. Core business logic should be implemented in independently testable modules.

## Configuration

Use environment variables for:

- LLM provider/model
- Hugging Face credentials
- Embedding model
- ChromaDB persistence path
- Chunk size/overlap
- Top-K retrieval

Never hard-code or commit secrets.

## Testing & Evaluation

Use pytest to test:

- document loading
- chunking
- embeddings
- ChromaDB retrieval
- RAG pipeline
- source attribution
- API endpoints
- error handling

Create a small evaluation dataset to measure retrieval quality and answer quality. Consider metrics such as Precision@K, Recall@K, MRR, answer correctness, context relevance, and groundedness.

## Development Approach

Develop incrementally:

1. Project foundation
2. Document ingestion
3. Chunking
4. Embeddings + ChromaDB
5. Semantic retrieval
6. RAG pipeline
7. Local LLM integration
8. Hugging Face integration
9. FastAPI APIs
10. Testing and evaluation
11. Documentation and cleanup

Before implementation, analyze the requirements and create an implementation plan. Implement and test each phase before proceeding to the next.

## Development Setup

```powershell
# One-time environment bootstrap
conda create -n rag python=3.12 -y
conda activate rag
pip install -r requirements.txt -r requirements-dev.txt

# Run the API locally
uvicorn app.main:app --reload   # then open http://127.0.0.1:8000/health

# Quality gates
pytest
ruff check .
```

Runtime settings are configured via environment variables; copy `.env.example` to `.env` for local overrides. Never commit `.env`.

## Expected Result

A complete end-to-end RAG system where users can upload documents, index them, ask questions through REST APIs, retrieve relevant context, receive grounded LLM-generated answers, and see the supporting document sources.
