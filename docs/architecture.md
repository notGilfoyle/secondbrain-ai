# Initial architecture

## Current request flows

Health: `HTTP request → health route → HealthService → response`

Ingestion: `multipart upload → document route → DocumentService → extractor → chunk_text → local file storage → SQLite document + chunks → metadata response`

The health and document routes obtain application services through FastAPI dependency injection. The document service receives plain Python values (filename, content type, and bounded file bytes); it does not depend on FastAPI request types.

## Code boundaries

- `api/`: HTTP routes and dependency providers.
- `application/`: use-case behavior independent of HTTP and infrastructure.
- `config/`: validated runtime settings sourced from `SB_` environment variables and an optional local `.env` file.
- `infrastructure/`: PDF/text extraction, local original-file storage, and SQLite persistence.
- `infrastructure/fastembed_provider.py`: optional local ONNX text embedding provider, independent of document ingestion.
- `main.py`: application factory and ASGI entry point.

`application/embeddings.py` defines the provider contract and cosine similarity calculation. The optional FastEmbed adapter loads `BAAI/bge-small-en-v1.5` once when an instance is constructed and batches texts through that model. Its first construction downloads and caches model files. Ingestion is independent of embedding generation.

`EmbeddingGenerationService` is an explicit use case for existing chunks; document ingestion does not call it. It loads chunks without an embedding for the provider's model identity, generates vectors in batches, validates finite numeric values and consistent dimensions, then persists them after inference completes. SQLite writes occur in one transaction, so inference failures create no rows and insertion failures roll back the full persistence batch. Retries skip chunks that already have rows for the same model.

`SQLiteChunkEmbeddingRepository` stores vectors as JSON arrays in `chunk_embeddings`, with a chunk foreign key and a composite primary key `(chunk_id, model_name)`. The separate `embedding_models` table stores the model identity and its fixed dimension; a composite foreign key ensures rows agree with that dimension. JSON is readable and simple for this prototype but larger and slower to decode than packed binary, and it does not support efficient vector search. A changed model revision should use a distinct identity. The use case currently accumulates all generated vectors in memory before the single database write transaction.

`SemanticRetrievalService` accepts a query and configurable positive `top_k` (default 5). It embeds the query through the same `EmbeddingProvider`, asks the repository for candidates with that exact model identity, and scores candidates with the shared cosine similarity utility. The SQLite repository uses one parameterized join across `chunk_embeddings`, `document_chunks`, and `documents`; query inference completes before the connection is opened. Results include chunk identity, order and text, document metadata, and cosine score, sorted descending. Missing embeddings are not candidates; malformed stored vectors, non-finite values, model identity mismatches, and incompatible dimensions raise errors.

This is exact semantic retrieval over stored vectors, not an LLM answer pipeline. The repository returns all stored candidate vectors for the selected model and ranking scans them in memory, with linear work in corpus vector count and size. It is intended for a small local corpus; no vector index or HTTP retrieval endpoint exists.

`RAGService` is an application-layer use case that composes a retriever with the `TextGenerationProvider` protocol. It rejects blank questions, retrieves ranked chunks, and skips generation when no context is found. For non-empty results it sends the question and JSON-encoded source metadata/text to the provider, with a system instruction to answer only from context, acknowledge insufficient evidence, treat retrieved text as untrusted data, and avoid invented facts or references. It returns generated text with deduplicated citations built from actual retrieval results (document ID, chunk ID, filename, and chunk index); the service does not accept citations from the model output. A blank model response raises an error, while retrieval and provider failures propagate.

Prompt-based grounding is advisory: an LLM can still ignore the instruction, produce unsupported claims, or fail to correctly interpret the context. The application does not independently verify the generated answer against its citations.

`CohereTextGenerationProvider` is an infrastructure adapter for Cohere's v2 Chat API. It sends system and user messages to the non-streaming `/v2/chat` endpoint using HTTPX; the application contract and `RAGService` remain vendor-independent. `SB_COHERE_API_KEY` and `SB_COHERE_MODEL` are optional settings and required only when constructing the provider. `SB_COHERE_TIMEOUT_SECONDS` defaults to 30 seconds. HTTP status, timeout, connection, malformed response, and empty output failures raise sanitized provider exceptions; request contents and credentials are not logged. HTTPX is included as a runtime dependency so the adapter can use its mock transport in deterministic tests.

To invoke RAG with Cohere, construct `SemanticRetrievalService` with the configured embedding provider and SQLite embedding repository, then inject it and `CohereTextGenerationProvider(settings)` into `RAGService`. The database must already contain vectors for that embedding model. There is no RAG HTTP route, streaming, retry, or conversation state. Production Cohere usage may be billed by input and output token counts, so retrieved context size and generated answer length affect cost; consult Cohere's current pricing information before live use.

The `documents` SQLite table stores a unique ID, original filename, canonical content type, safe relative storage filename, size, extracted text, and UTC creation timestamp. The `document_chunks` table stores unique chunk IDs, source document IDs, zero-based indexes, text, and creation timestamps. A foreign key with cascade delete and a unique `(document_id, chunk_index)` constraint preserve ownership and ordering. Connections enable foreign-key enforcement. Schema initialization uses `CREATE TABLE IF NOT EXISTS`, so existing databases retain their document data as the new table is added. Repository calls use parameterized SQL.

Ingestion calls the standalone `chunk_text` utility with its 1000-character size and 150-character overlap defaults. The SQLite repository inserts the document and all chunks in one transaction; a chunk insertion failure rolls the document insert back as well. Original files are written to a temporary file in the configured storage directory and linked to a server-generated UUID filename without overwriting an existing path.

If the database insert fails after the file is written, the application attempts to delete that file. This is compensation, not a distributed transaction: a process crash can still leave an orphan file, and a cleanup failure can leave one behind. Files are not deleted when a database row is manually removed.

The configured default upload file limit is 10 MiB. A pure ASGI receive middleware caps the complete multipart request at that limit plus 64 KiB for boundaries and headers, counting streamed bytes even when `Content-Length` is missing or too small. A valid oversized `Content-Length` is rejected before reading the body. On streamed overflow the middleware raises Starlette's multipart parsing exception so partially spooled upload files are closed, and the app maps that specific parser error to HTTP 413. The body is passed through once and is not buffered or replayed by the middleware. The ASGI server allocates each received message before middleware can inspect it, so a single oversized ASGI message may already occupy memory; the middleware does not accumulate the whole request.

## Offline retrieval evaluation

`secondbrain.evaluation` contains a small offline metric runner for the
retrieval contract. `evaluation/dataset.json` is synthetic and version
controlled; `venv/bin/python scripts/evaluate_retrieval.py --k 3` injects fixed
vectors and a fake candidate repository into the actual
`SemanticRetrievalService`. It reports Recall@k, mean reciprocal rank (MRR),
and answerable queries with no relevant chunk in the top k. Recall@k is the
share of expected supporting chunk IDs found within k, averaged per answerable
query; MRR averages the reciprocal rank of each query's first relevant chunk.
Unanswerable cases are excluded from these ranking metrics, so this evaluation
are also excluded from retrieval false-positive evaluation, so it does not
measure irrelevant candidates returned for those questions or test abstention.
Fixed vectors test plumbing and metric behavior, not
the semantic quality of FastEmbed vectors. LLM answer quality is not measured
by this offline suite; prompt instructions alone do not ensure factual
grounding. See `docs/evaluation-report.md` for the recorded baseline and
limitations.

## Future direction

OCR-integrated automatic embedding, vector indexing, authentication, and cloud storage are not implemented. Character-based chunks and explicitly generated embeddings are persisted locally in SQLite. Semantic retrieval and RAG are application use cases without HTTP endpoints. Retrieval evaluation currently uses a small synthetic fixture; broader corpus coverage and separate, human-reviewed live answer-quality evaluation are future work.
