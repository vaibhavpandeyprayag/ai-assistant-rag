# AI Assistant with RAG

## Overview

Build a modular AI Assistant that uses Retrieval-Augmented Generation (RAG) to process user-provided documents and answer questions using relevant retrieved context.

The system will use:

- Python
- FastAPI
- LangChain
- ChromaDB
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

Uses the Hugging Face Inference API for generation, configured via `HUGGINGFACE_API_KEY` and `HUGGINGFACE_MODEL` (e.g. `meta-llama/Llama-3.1-8B-Instruct`).

LLM selection is configuration-driven so the RAG pipeline does not depend on a hard-coded model.

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
- Allowed CORS origins

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
7. Hugging Face LLM integration
8. FastAPI APIs
9. Testing and evaluation
10. Documentation and cleanup

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

`torch` + `sentence-transformers` (local embeddings) install only on Windows
via requirements markers. On other platforms use `EMBEDDING_PROVIDER=hf` (see
the Render deployment section).

Runtime settings are configured via environment variables; copy `.env.example` to `.env` for local overrides. Never commit `.env`.

## Deployment (Render)

The repo ships a Render Blueprint (`render.yaml`). The backend deploys as a
Python web service; the Vite client is deployed separately as a static site
(its `CORS_ORIGINS` must include the client's URL).

Key points:

- **Ephemeral storage**: the free plan has no persistent disk. `data/`
  (uploads and the Chroma index) is wiped on every restart/redeploy, so
  documents must be re-uploaded and re-ingested after each deploy.
- **Lightweight Linux build**: `torch` and `sentence-transformers` are
  Windows-only in `requirements.txt`. Render uses `EMBEDDING_PROVIDER=hf`, so
  embeddings are served by the Hugging Face Inference API and the CUDA torch
  wheels (~2.5 GB) are never installed.
- **`HUGGINGFACE_API_KEY`** is not part of the blueprint (`sync: false`); set
  it in the Render dashboard after the first deploy.

Deploy steps: push to GitHub, then in Render "New +" → Blueprint, select the
repo, and deploy. The health check hits `GET /health`.

## Expected Result

A complete end-to-end RAG system where users can upload documents, index them, ask questions through REST APIs, retrieve relevant context, receive grounded LLM-generated answers, and see the supporting document sources.
