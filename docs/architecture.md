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

`application/embeddings.py` defines the provider contract and cosine similarity calculation. The optional FastEmbed adapter loads `BAAI/bge-small-en-v1.5` once when an instance is constructed and batches texts through that model. Its first construction downloads and caches model files. Embeddings remain in memory; they are not stored in SQLite or connected to ingestion.

`EmbeddingGenerationService` is an explicit use case for existing chunks; document ingestion does not call it. It loads chunks without an embedding for the provider's model identity, generates vectors in batches, validates finite numeric values and consistent dimensions, then persists them after inference completes. SQLite writes occur in one transaction, so inference failures create no rows and insertion failures roll back the full persistence batch. Retries skip chunks that already have rows for the same model.

`SQLiteChunkEmbeddingRepository` stores vectors as JSON arrays in `chunk_embeddings`, with a chunk foreign key and a composite primary key `(chunk_id, model_name)`. The separate `embedding_models` table stores the model identity and its fixed dimension; a composite foreign key ensures rows agree with that dimension. JSON is readable and simple for this prototype but larger and slower to decode than packed binary, and it does not support efficient vector search. A changed model revision should use a distinct identity. The use case currently accumulates all generated vectors in memory before the single database write transaction.

The `documents` SQLite table stores a unique ID, original filename, canonical content type, safe relative storage filename, size, extracted text, and UTC creation timestamp. The `document_chunks` table stores unique chunk IDs, source document IDs, zero-based indexes, text, and creation timestamps. A foreign key with cascade delete and a unique `(document_id, chunk_index)` constraint preserve ownership and ordering. Connections enable foreign-key enforcement. Schema initialization uses `CREATE TABLE IF NOT EXISTS`, so existing databases retain their document data as the new table is added. Repository calls use parameterized SQL.

Ingestion calls the standalone `chunk_text` utility with its 1000-character size and 150-character overlap defaults. The SQLite repository inserts the document and all chunks in one transaction; a chunk insertion failure rolls the document insert back as well. Original files are written to a temporary file in the configured storage directory and linked to a server-generated UUID filename without overwriting an existing path.

If the database insert fails after the file is written, the application attempts to delete that file. This is compensation, not a distributed transaction: a process crash can still leave an orphan file, and a cleanup failure can leave one behind. Files are not deleted when a database row is manually removed.

The configured default upload file limit is 10 MiB. A pure ASGI receive middleware caps the complete multipart request at that limit plus 64 KiB for boundaries and headers, counting streamed bytes even when `Content-Length` is missing or too small. A valid oversized `Content-Length` is rejected before reading the body. On streamed overflow the middleware raises Starlette's multipart parsing exception so partially spooled upload files are closed, and the app maps that specific parser error to HTTP 413. The body is passed through once and is not buffered or replayed by the middleware. The ASGI server allocates each received message before middleware can inspect it, so a single oversized ASGI message may already occupy memory; the middleware does not accumulate the whole request.

## Future direction

OCR-integrated automatic embedding, similarity search, RAG, authentication, and cloud storage are not implemented. Character-based chunks and explicitly generated embeddings are persisted locally in SQLite, but there is no vector index or search endpoint.
