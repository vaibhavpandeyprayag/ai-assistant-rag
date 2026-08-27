# AGENTS.md

## Role

You are the coding agent for this AI Assistant with RAG project.

Your job is to analyze requirements, plan carefully, implement incrementally, test changes, and maintain a clean, modular codebase.

The project requirements are defined in `README.md`. Treat it as the primary source of product requirements.

---

## Development Principles

- Analyze the existing project before making changes.
- Do not implement the entire project in one step unless explicitly requested.
- Prefer incremental implementation by phase or feature.
- Keep architecture modular and maintainable.
- Avoid unnecessary abstractions and dependencies.
- Do not rewrite working code without a clear reason.
- Preserve existing behavior unless a requirement explicitly changes it.
- When requirements are ambiguous and the ambiguity materially affects architecture or behavior, ask for clarification rather than guessing.
- Explain important technical trade-offs when proposing architectural decisions.

---

## Coding Style

### Python

- Use modern, readable Python.
- Use type hints for functions, methods, and important variables.
- Follow PEP 8 and standard Python conventions.
- Provide proper Python comments
- Prefer clear names over abbreviations.
- Keep functions and classes focused on one responsibility.
- Avoid deeply nested logic.
- Use appropriate exception handling.
- Avoid broad `except Exception` unless there is a clear reason and the error is handled appropriately.
- Prefer explicit, maintainable code over clever code.

### FastAPI

- Keep API route handlers thin.
- Do not put core business logic directly inside route handlers.
- Use Pydantic models for request and response schemas.
- Use dependency injection where it improves testability or separation of concerns.
- Return appropriate HTTP status codes.
- Validate user input.
- Do not expose internal stack traces or secrets through API responses.

### Project Structure

Separate concerns such as:

- API/routes
- configuration
- document ingestion
- text processing/chunking
- embeddings
- vector database
- retrieval
- prompts
- LLM integration
- RAG orchestration
- application services
- schemas
- tests

Do not put the entire application into a single module.

---

## RAG Architecture

The intended document pipeline is:

```text
Document
    ↓
Document Loader
    ↓
Text Extraction
    ↓
Text Preprocessing
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB
```

The intended question-answering pipeline is:

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

Keep document ingestion and query-time retrieval logically separate.

---

## Document Processing

Initial supported formats:

- PDF
- DOCX
- TXT
- Markdown

The ingestion pipeline should:

1. Load the document.
2. Extract text.
3. Preserve useful metadata.
4. Perform appropriate text cleaning.
5. Split text into chunks.
6. Generate embeddings.
7. Store chunks, embeddings, and metadata in ChromaDB.

At minimum, preserve metadata such as:

- `document_id`
- `filename`
- `source`
- `chunk_id`
- `page_number` where available

Design document processing so additional formats can be added without major architectural changes.

---

## Chunking

Chunking must be configurable.

At minimum, support configuration for:

- chunk size
- chunk overlap
- chunking strategy

Do not hard-code these values throughout the codebase.

Choose sensible initial defaults during implementation and document the reasoning when relevant.

---

## Embeddings

Use an embedding abstraction rather than coupling the application to one embedding model.

The embedding model should be configurable.

The same embedding model/configuration must be used consistently for document chunks and user queries.

A suitable Hugging Face/Sentence Transformers embedding model may be used initially.

---

## ChromaDB

Use ChromaDB as the initial vector database.

It should support:

- persistent local storage
- vector similarity search
- configurable Top-K retrieval
- metadata storage
- metadata filtering where appropriate

Keep ChromaDB-specific logic isolated behind a service/repository abstraction so the vector store could be replaced later.

---

## Retrieval

The retrieval layer should:

1. Convert the user query into an embedding.
2. Search ChromaDB.
3. Retrieve the configured Top-K chunks.
4. Preserve source metadata.
5. Return structured retrieval results to the RAG layer.

Retrieval should be independently testable without requiring an LLM.

Avoid retrieving excessive context unnecessarily.

---

## RAG Prompting

The RAG prompt should instruct the LLM to:

- answer using the supplied retrieved context
- prioritize retrieved information
- avoid inventing unsupported facts
- acknowledge when the context is insufficient
- provide source information where applicable

Keep prompts separate from application/business logic so they can be modified and evaluated independently.

---

## LLM Architecture

The initial LLM provider is the Hugging Face model/API (Inference API).

The RAG pipeline should interact with an LLM abstraction rather than directly depending on one concrete provider.

Switching the chosen model/API should primarily require configuration changes.

Never hard-code API keys or credentials.

Use environment variables and provide configuration through `.env.example`.

---

## FastAPI API

Initial API capabilities should include:

```text
GET  /health
POST /documents/upload
POST /documents/ingest
POST /search
POST /chat
```

The exact request/response schemas should be determined during the planning phase.

The chat response should be capable of returning:

- generated answer
- supporting sources
- useful source metadata

Do not unnecessarily expose internal implementation details.

---

## Configuration

Centralize application configuration.

Potential configuration values include:

```text
LLM_PROVIDER
HUGGINGFACE_API_KEY
HUGGINGFACE_MODEL
EMBEDDING_MODEL
CHROMA_PERSIST_DIRECTORY
CHUNK_SIZE
CHUNK_OVERLAP
TOP_K
```

Use environment variables for configurable runtime settings.

Never commit real credentials, API keys, or secrets.

---

## Testing

Use pytest.

Tests should cover, as appropriate:

### Document Processing

- PDF loading
- DOCX loading
- TXT loading
- Markdown loading
- text extraction
- chunking
- metadata preservation

### Retrieval

- embedding generation
- ChromaDB storage
- similarity search
- Top-K retrieval
- metadata filtering

### RAG

- context construction
- prompt construction
- LLM integration
- source attribution
- insufficient-context behavior

### API

- health endpoint
- document upload
- ingestion
- search
- chat
- validation
- error handling

Do not consider a feature complete merely because its implementation runs; add appropriate automated tests.

---

## Evaluation

The project should eventually include a small evaluation dataset containing:

- user question
- expected answer
- expected source document

Evaluate retrieval and generation separately where practical.

Potential retrieval metrics:

- Precision@K
- Recall@K
- MRR

Potential generation/system metrics:

- answer correctness
- context relevance
- faithfulness/groundedness
- unsupported answer rate
- retrieval latency
- end-to-end latency

Do not add complicated evaluation infrastructure prematurely. Start with a small reproducible evaluation setup.

---

## Error Handling

Handle common failures gracefully:

- unsupported file format
- corrupted document
- empty document
- empty query
- no relevant results
- embedding failure
- ChromaDB failure
- LLM failure
- invalid API input
- missing configuration
- external API/authentication failure

Return useful errors without exposing secrets, stack traces, or unnecessary internal details.

---

## Security

- Never hard-code secrets.
- Never commit `.env`.
- Validate uploaded files.
- Apply reasonable file-size/input limits.
- Validate API input.
- Do not execute arbitrary content from uploaded documents.
- Avoid exposing internal system details in API responses.

---

## Dependencies

Use only dependencies that provide clear value.

Before adding a dependency:

1. Check whether existing dependencies already provide the required capability.
2. Prefer stable and actively maintained libraries.
3. Avoid unnecessary overlapping libraries.
4. Consider compatibility with the current Python and LangChain ecosystem.

Use current, supported APIs and avoid deprecated LangChain interfaces.

---

## Agent Workflow

When asked to work on this project:

### For planning tasks

1. Read `README.md`.
2. Read `AGENTS.md`.
3. Inspect the existing repository.
4. Identify requirements, constraints, dependencies, and risks.
5. Propose the architecture and implementation plan.
6. Break implementation into logical phases.
7. Do not write implementation code unless explicitly requested.

### For implementation tasks

1. Read `README.md` and `AGENTS.md`.
2. Inspect relevant existing code.
3. Confirm which phase/feature is being implemented.
4. Make the smallest coherent set of changes.
5. Run relevant tests.
6. Fix failures caused by the changes.
7. Review the resulting code for consistency.
8. Update documentation when behavior or setup changes.
9. Summarize what changed and what remains.

### For debugging tasks

1. Reproduce or inspect the reported problem.
2. Identify the root cause before changing code.
3. Avoid unrelated refactoring.
4. Implement the smallest appropriate fix.
5. Add or update a regression test where practical.
6. Run the relevant test suite.

---

## Planning Expectations

The first major task is to create a sound implementation plan for the RAG system.

The plan should consider:

- project structure
- component boundaries
- document loaders
- chunking strategy
- embedding model
- ChromaDB design
- retrieval strategy
- metadata design
- RAG prompt design
- LLM abstraction
- Hugging Face integration
- FastAPI API design
- configuration
- testing
- evaluation
- error handling
- security
- future extensibility

Do not prematurely implement features while planning.

Where multiple technically valid approaches exist, compare them briefly and recommend one based on simplicity, maintainability, compatibility, and suitability for this project.

---

## Scope Control

The project is primarily a RAG-based AI Assistant.

Do not introduce unrelated features such as:

- autonomous multi-agent systems
- web browsing
- complex long-term memory
- voice interfaces
- authentication systems
- production cloud infrastructure
- unnecessary frontend development

unless explicitly requested.

These can be considered future extensions rather than part of the initial implementation.

---

## Definition of Done

A feature is considered complete when:

- the implementation satisfies its requirement
- the code follows the project's architecture and style
- relevant tests exist
- relevant tests pass
- configuration is handled appropriately
- errors are handled reasonably
- documentation is updated when necessary

The final objective is a clean, modular, testable RAG application rather than merely a working prototype.
