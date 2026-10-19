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
- `main.py`: application factory and ASGI entry point.

The `documents` SQLite table stores a unique ID, original filename, canonical content type, safe relative storage filename, size, extracted text, and UTC creation timestamp. The `document_chunks` table stores unique chunk IDs, source document IDs, zero-based indexes, text, and creation timestamps. A foreign key with cascade delete and a unique `(document_id, chunk_index)` constraint preserve ownership and ordering. Connections enable foreign-key enforcement. Schema initialization uses `CREATE TABLE IF NOT EXISTS`, so existing databases retain their document data as the new table is added. Repository calls use parameterized SQL.

Ingestion calls the standalone `chunk_text` utility with its 1000-character size and 150-character overlap defaults. The SQLite repository inserts the document and all chunks in one transaction; a chunk insertion failure rolls the document insert back as well. Original files are written to a temporary file in the configured storage directory and linked to a server-generated UUID filename without overwriting an existing path.

If the database insert fails after the file is written, the application attempts to delete that file. This is compensation, not a distributed transaction: a process crash can still leave an orphan file, and a cleanup failure can leave one behind. Files are not deleted when a database row is manually removed.

The configured default upload file limit is 10 MiB. A pure ASGI receive middleware caps the complete multipart request at that limit plus 64 KiB for boundaries and headers, counting streamed bytes even when `Content-Length` is missing or too small. A valid oversized `Content-Length` is rejected before reading the body. On streamed overflow the middleware raises Starlette's multipart parsing exception so partially spooled upload files are closed, and the app maps that specific parser error to HTTP 413. The body is passed through once and is not buffered or replayed by the middleware. The ASGI server allocates each received message before middleware can inspect it, so a single oversized ASGI message may already occupy memory; the middleware does not accumulate the whole request.

## Future direction

OCR, embeddings, semantic search, RAG, authentication, and cloud storage are not implemented. Character-based chunks are persisted locally in SQLite, but there is no chunk retrieval API or semantic ranking.
