# Implementation Plan — AI Assistant with RAG

> Planning deliverable derived from `README.md` (primary product requirements) and `agents.md` (engineering principles).
> Status: approved. Confirmed decisions are marked **[confirmed]**.

---

## 1. Requirements Summary

- **Stack**: Python, FastAPI, LangChain, ChromaDB, Pydantic, pytest; Local LLMs + Hugging Face
- **Ingestion**: PDF / DOCX / TXT / MD → extract → clean → chunk → embed → ChromaDB, preserving `document_id`, `filename`, `source`, `chunk_id`, `page_number`
- **QA pipeline**: query → embed → top-K similarity search → context → grounded prompt → LLM → answer + sources
- **API**: `/health`, `/documents/upload`, `/documents/ingest`, `/search`, `/chat`
- **Cross-cutting**: env-based config (no secrets in code), typed error handling, pytest coverage per layer, small evaluation harness

## 2. Environment Constraints (observed)

- Active interpreter was Python 3.9.25 (miniconda base, EOL) → project targets a dedicated conda env with Python 3.12. **[confirmed]**
- Ollama is installed locally → used for the "Local LLM" requirement (separate process, no C++ compilation issues on Windows, trivially mockable HTTP boundary).
- Only `pip` available (no uv/poetry) → plain `requirements.txt` / `requirements-dev.txt`.

## 3. Proposed Architecture

```
app/
  main.py                 # app factory, routers, exception handlers
  config.py               # Settings (pydantic-settings, .env + env vars)
  errors.py               # domain exceptions → HTTP mapping
  api/
    dependencies.py       # cached singletons (DI): settings, embedder, repo, llm, services
    routes/{health,documents,search,chat}.py
  schemas/{documents,search,chat}.py
  ingestion/
    models.py             # LoadedDocument, Section, Chunk dataclasses
    loaders.py            # pypdf / python-docx / txt / md dispatch
    cleaning.py           # whitespace/control-char normalization
    chunking.py           # configurable chunker (size, overlap, strategy)
  embeddings/             # EmbeddingProvider protocol + local ST + HF-API impls + factory
  vectordb/               # VectorStoreRepository protocol + ChromaPersistentRepository
  retrieval/retriever.py  # query → embed → top-K → RetrievedChunk[]
  llm/                    # LLMProvider protocol + Ollama + HF impls + factory
  rag/                    # prompts.py, context_builder.py, pipeline.py
  services/               # ingestion_service.py, chat_service.py (orchestration)
data/{uploads,chroma}/    # gitignored runtime artifacts
evaluation/               # dataset.jsonl, sample_docs/, metrics.py, eval scripts
tests/{unit,integration,api}
```

### Key Design Decisions & Trade-offs

| Decision | Recommendation | Rationale / alternative |
|---|---|---|
| LangChain depth | **Lean** **[confirmed]**: only `langchain-text-splitters` for chunking; direct `pypdf`, `python-docx`, `sentence-transformers`, `chromadb`, `httpx`, `huggingface_hub` behind 3 small internal Protocols | Full LangChain partner packages reduce custom code but add dependency churn and opaque behavior; direct libs keep page-level metadata control and stable APIs. Keeps README's LangChain requirement satisfied where it adds value |
| Local LLM | **Ollama via HTTP** (`httpx`) | Already installed; avoids llama-cpp-python compilation pain on Windows; transformers-pipeline is too slow for generation |
| HF integration | **Inference API only** **[confirmed]** via `huggingface_hub.InferenceClient` | "models/APIs" — API path is the distinct second provider; local HF inference duplicates the Ollama role |
| Upload vs ingest split | Upload validates + persists raw file (fast); ingest processes stored files (slow, re-runnable) | Decouples embedding cost from upload; enables re-ingest with different chunk params |
| Sync route handlers | Yes (FastAPI threadpool) | Workload is CPU/blocking-IO bound; avoids async complexity |
| Abstractions | Exactly 3 Protocols: `EmbeddingProvider`, `VectorStoreRepository`, `LLMProvider` | Required by AGENTS.md (swappable store/provider, LLM-free retrieval tests); no more than needed |
| Chunk defaults | `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, strategy=`recursive` | ~1000 chars ≈ fits MiniLM's 256-token window; ~20% overlap preserves continuity |
| `document_id` | UUID generated at upload; ingest deletes prior chunks for that ID (idempotent re-ingest) | Alternative (content-hash IDs) merges same-content files — surprising behavior |

### Metadata stored per chunk

Chroma-safe types only (str/int/float/bool; Chroma rejects nulls):

- `document_id` (str)
- `filename` (str)
- `source` (str)
- `chunk_id` (str)
- `chunk_index` (int)
- `page_number` (int, `-1` when unavailable)

### API Contracts (draft)

```
GET  /health              → {status, version, llm_provider, embedding_model}
POST /documents/upload    (multipart) → 201 {document_id, filename, size_bytes, status:"stored"}
POST /documents/ingest    {filenames?: [..]}   → {results: [{filename, document_id, n_chunks, status, error?}]}
POST /search              {query, top_k?, filter?: {document_id?|filename?}} → {results: [{text, score, metadata}]}
POST /chat                {query, top_k?} → {answer, sources[], insufficient_context, latency_ms}
Errors: {"code": "...", "message": "..."} — 404/413/415/422/500/503; never stack traces or secrets
```

### Configuration Keys

```
LLM_PROVIDER            # local | hf
LOCAL_LLM_MODEL         # e.g. llama3.1:8b (Ollama tag)
HUGGINGFACE_API_KEY     # secret, env only
HUGGINGFACE_MODEL       # HF Inference chat model id
EMBEDDING_MODEL         # default sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY# default ./data/chroma
UPLOAD_DIRECTORY        # default ./data/uploads
CHUNK_SIZE              # default 1000
CHUNK_OVERLAP           # default 200
CHUNK_STRATEGY          # recursive | fixed
TOP_K                   # default 5
RETRIEVAL_MIN_SCORE     # default disabled (0.0)
UPLOAD_MAX_SIZE_MB      # default ~20
```

---

## 4. Phases

### Phase 1 — Foundation, configuration & health endpoint

1. **Objective**: Runnable skeleton with centralized config and test tooling.
2. **Requirements**: Package layout; `Settings` via pydantic-settings covering all keys (incl. `UPLOAD_MAX_SIZE_MB`, `RETRIEVAL_MIN_SCORE`); `.env.example`; `.gitignore` (`data/`, `.env`); logging setup; app factory + `GET /health` (no secrets in response); ruff config.
3. **Files**: `app/main.py`, `app/config.py`, `app/errors.py`, `app/api/routes/health.py`, `app/api/dependencies.py`, `requirements*.txt`, `.env.example`, `.gitignore`, `tests/unit/test_config.py`, `tests/api/test_health.py`.
4. **Dependencies**: none.
5. **Testing**: health response shape; defaults + env overrides; API keys never serialized.
6. **Acceptance**: `uvicorn app.main:app` serves `/health`; `pytest` green; `ruff check` clean.

### Phase 2 — Document loading & text extraction

1. **Objective**: Format-specific extractors producing `LoadedDocument` (sections with page info).
2. **Requirements**: PDF per-page via `pypdf`; DOCX paragraphs via `python-docx`; TXT/MD UTF-8 read; dispatch by extension; typed errors: `UnsupportedFormatError`, `DocumentParseError` (corrupt), `EmptyDocumentError`.
3. **Files**: `app/ingestion/models.py`, `app/ingestion/loaders.py`, `tests/unit/test_loaders_*.py`, `tests/fixtures/`.
4. **Dependencies**: Phase 1.
5. **Testing**: extraction correctness ×4 formats; PDF page numbers preserved; corrupt/unsupported/empty → correct error types.
6. **Acceptance**: All formats load with expected structure; no raw third-party stack traces escape.

### Phase 3 — Cleaning & configurable chunking

1. **Objective**: Deterministic preprocessing + overlap-aware splitting into `Chunk` records.
2. **Requirements**: Conservative cleaner (collapse whitespace, strip control chars); `Chunker` with `strategy ∈ {recursive, fixed}` (recursive via `langchain-text-splitters`), size/overlap from config; `chunk_id = "{document_id}::{index}"`; inherits `page_number` (start page); validate `overlap < size`.
3. **Files**: `app/ingestion/cleaning.py`, `app/ingestion/chunking.py`, `tests/unit/test_cleaning.py`, `tests/unit/test_chunking.py`.
4. **Dependencies**: Phase 2.
5. **Testing**: size limits respected; overlap across boundaries; page attribution; empty input → `[]`; invalid config raises.
6. **Acceptance**: Defaults documented with rationale; pure functions fully unit-tested.

### Phase 4 — Embedding abstraction

1. **Objective**: Swappable `EmbeddingProvider`; same model guaranteed for docs and queries.
2. **Requirements**: Protocol (`embed_documents`, `embed_query`); `SentenceTransformersProvider` (default `all-MiniLM-L6-v2`, batched, normalized); optional `HuggingFaceInferenceProvider` (feature-extraction, key from env); factory on `EMBEDDING_PROVIDER`; lazy model load.
3. **Files**: `app/embeddings/{base,local,hf_api,factory}.py`, `tests/unit/test_embeddings_factory.py`, `tests/integration/test_embeddings_local.py` (marked slow).
4. **Dependencies**: Phase 1.
5. **Testing**: doc/query dimension consistency; batching; network tests opt-in via marker.
6. **Acceptance**: Single injected instance used by ingestion *and* retrieval; provider switch = config change.

### Phase 5 — ChromaDB vector store repository

1. **Objective**: Persistence/search isolated behind `VectorStoreRepository`.
2. **Requirements**: `chromadb.PersistentClient` at `CHROMA_PERSIST_DIRECTORY`; collection `documents`, cosine space; `add_chunks` (upsert), `similarity_search(query_embedding, top_k, where)`, `delete_document(document_id)`, `count()`; similarity = `1 − distance`; metadata type whitelist; filters on `document_id`/`filename`.
3. **Files**: `app/vectordb/{base,chroma}.py`, `tests/integration/test_chroma_repository.py` (tmp_path).
4. **Dependencies**: Phase 3 (Chunk model); Phase 4 conceptually (tests use synthetic vectors).
5. **Testing**: add/query roundtrip + ordering; top-k; filters; delete-by-document; persistence across reopen; dimension mismatch → typed error.
6. **Acceptance**: No LangChain/LLM imports in this layer; store replaceable per AGENTS.md.

### Phase 6 — Ingestion service (end-to-end indexing)

1. **Objective**: Orchestrate load → clean → chunk → embed → store with idempotent upsert.
2. **Requirements**: `ingest_file` → `{document_id, filename, n_chunks, status}`; delete-then-insert per `document_id`; batch continues past individual failures; rejects empty docs; timings logged.
3. **Files**: `app/services/ingestion_service.py`, `tests/integration/test_ingestion_service.py` (fake embedder + tmp Chroma).
4. **Dependencies**: Phases 2–5.
5. **Testing**: happy path; re-ingest produces no duplicates; partial batch failure isolation; empty/corrupt handling.
6. **Acceptance**: Populated index with complete metadata; testable without LLM/network.

### Phase 7 — Retrieval service

1. **Objective**: Query-time top-K retrieval, independent of any LLM.
2. **Requirements**: `retrieve(query, top_k?, filter?)` → `RetrievedChunk{text, score, metadata}[]`; shares the configured embedder; optional `RETRIEVAL_MIN_SCORE` (default off); empty query/index handled explicitly.
3. **Files**: `app/retrieval/retriever.py`, `tests/integration/test_retriever.py`.
4. **Dependencies**: Phases 4, 5.
5. **Testing**: ranking sanity with fake embedder; top-k/filter/threshold behavior; empty-index path.
6. **Acceptance**: Satisfies AGENTS.md "retrieval independently testable without requiring an LLM".

### Phase 8 — LLM abstraction (Ollama + Hugging Face)

1. **Objective**: Provider-neutral generation; switching = env change only.
2. **Requirements**: `LLMProvider.generate(messages, options) -> str`; `OllamaLLM` (httpx → `localhost:11434`, connection failure → `LLMUnavailableError`); `HuggingFaceLLM` (`InferenceClient.chat_completion`, auth failure → `LLMAuthError`); factory on `LLM_PROVIDER ∈ {local, hf}`; `FakeLLMProvider` test utility; secrets never logged.
3. **Files**: `app/llm/{base,ollama_llm,hf_llm,factory}.py`, `tests/utils/fakes.py`, `tests/unit/test_llm_*.py` (httpx `MockTransport` — no extra mocking dep).
4. **Dependencies**: Phase 1.
5. **Testing**: request building; timeout/error mapping; factory selection; log redaction.
6. **Acceptance**: Both providers behind one interface; failures typed and safe.

### Phase 9 — RAG pipeline

1. **Objective**: Grounded QA orchestration returning answer + sources.
2. **Requirements**: `ContextBuilder` (numbered `[1] (file, p.X)` blocks, char budget); templates in `prompts.py` enforcing: answer only from context, cite sources, declare insufficiency; `RAGPipeline.answer()` → `{answer, sources[], retrieved_chunks[], insufficient_context, latency_ms}`; short-circuit canned "insufficient context" response (no LLM call) when nothing retrieved.
3. **Files**: `app/rag/{prompts,context_builder,pipeline}.py`, `tests/unit/test_context_builder.py`, `tests/unit/test_prompts.py`, `tests/integration/test_rag_pipeline.py` (fake retriever + fake LLM).
4. **Dependencies**: Phases 7, 8.
5. **Testing**: context truncation; prompt contains grounding rules + all chunk refs; insufficient path skips LLM; sources ↔ chunks 1:1; latencies recorded.
6. **Acceptance**: Deterministic end-to-end with fakes; grounding rules verifiable in rendered prompt.

### Phase 10 — REST API endpoints & error handling

1. **Objective**: Thin routes exposing the pipelines with validated schemas and safe errors.
2. **Requirements**: Endpoints per contract above; upload validation (extension whitelist, size cap, sanitized filenames); global exception handlers (domain → 4xx JSON; unexpected → generic 500, logged server-side); DI via cached providers; lifespan cleanup of HTTP clients.
3. **Files**: `app/api/routes/{documents,search,chat}.py`, `app/api/dependencies.py`, `app/schemas/*.py`, `app/services/chat_service.py`, `app/main.py`, `tests/api/*`.
4. **Dependencies**: Phases 6, 7, 9.
5. **Testing**: Full matrix — bad extension 415, oversize 413, empty query 422, unknown file 404, chat insufficient-context shape, secret-free error bodies.
6. **Acceptance**: Contract implemented; no internals leaked; limits enforced.

### Phase 11 — Evaluation harness (minimal)

1. **Objective**: Reproducible measurement of retrieval and generation quality.
2. **Requirements**: `dataset.jsonl` (~15 items: question, expected answer, source file); tiny sample corpus; pure-function metrics: Precision@K, Recall@K, MRR; retrieval eval script; generation eval reporting cited-source match (groundedness proxy) + latency. No LLM-as-judge infrastructure yet.
3. **Files**: `evaluation/{dataset.jsonl, sample_docs/, metrics.py, evaluate_retrieval.py, evaluate_generation.py}`, `tests/unit/test_metrics.py`.
4. **Dependencies**: Phases 6, 7, 9.
5. **Testing**: Metric math asserted against synthetic rankings.
6. **Acceptance**: Two commands print deterministic metrics; math unit-tested.

### Phase 12 — Documentation & polish

1. **Objective**: Hand-off quality; fresh-machine reproducibility.
2. **Requirements**: README: setup (conda env, Python ≥3.11), `ollama pull <model>`, env var reference, curl examples, architecture diagram, eval usage; finalize `.env.example`; lint clean; dead code removed.
3. **Files**: `README.md`, `.env.example`, minor refactors.
4. **Dependencies**: all.
5. **Testing**: full suite green; manual smoke test following README only.
6. **Acceptance**: New machine reaches end-to-end chat using README alone.

---

## 5. Recommended Phase Order

```
P1 Foundation
 ├─ P2 Loaders → P3 Chunking ─┐
 ├─ P4 Embeddings ────────────┼─→ P5 Chroma repo → P6 Ingestion ─┐
 └─ P8 LLM providers ─────────┴─→ P7 Retriever ──→ P9 RAG ───────┼→ P10 API → P12 Docs
                                                                 └→ P11 Evaluation
```

Sequential execution P1 → P12 works as-is; P4 and P8 are independent and can be pulled earlier/in parallel after P1.

| # | Phase | Depends on |
|---|-------|-----------|
| 1 | Foundation: layout, Settings, `/health`, tooling | — |
| 2 | Document loaders (PDF/DOCX/TXT/MD) + typed errors | 1 |
| 3 | Cleaning + configurable chunking | 2 |
| 4 | Embedding providers (local ST + HF API) | 1 |
| 5 | ChromaDB repository behind protocol | 3, 4 |
| 6 | Ingestion service (idempotent end-to-end indexing) | 2–5 |
| 7 | Retrieval service (LLM-free) | 4, 5 |
| 8 | LLM providers (Ollama + HF API) + factory | 1 |
| 9 | RAG pipeline (context builder, prompts, orchestration) | 7, 8 |
| 10 | REST API (upload/ingest/search/chat) + error handling | 6, 7, 9 |
| 11 | Evaluation harness (P@K, R@K, MRR + groundedness proxy) | 6, 7, 9 |
| 12 | Documentation & polish | all |

## 6. Dependency Set

- **Runtime**: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `python-multipart`, `pypdf`, `python-docx`, `langchain-text-splitters`, `sentence-transformers`, `chromadb`, `httpx`, `huggingface_hub`
- **Dev**: `pytest`, `ruff` (httpx `MockTransport` covers HTTP mocking — no extra lib)

## 7. Confirmed Decisions

1. **LangChain depth**: Lean — only `langchain-text-splitters`; direct libraries elsewhere behind 3 internal Protocols.
2. **Python environment**: New conda env (`rag`) with Python 3.12; active 3.9 base is EOL.
3. **Hugging Face scope**: Inference API only (`InferenceClient` + `HUGGINGFACE_API_KEY`); local LLM role covered by Ollama.
