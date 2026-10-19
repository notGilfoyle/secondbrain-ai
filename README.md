# SecondBrain AI

A personal knowledge management application. The current vertical slice accepts PDF, Markdown, and plain-text files, extracts their text, stores metadata and text in SQLite, retains the original files locally, and lists ingested documents.

## Local setup

Requires Python 3.11 through 3.14.

```bash
python3 -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
uvicorn secondbrain.main:app --reload --app-dir src
```

The API is available at `http://127.0.0.1:8000`; the health endpoint is `GET /health`, document endpoints are under `/api/v1/documents`, and the OpenAPI UI is `/docs`.

Configuration uses the `SB_` environment-variable prefix. Local `.env` values are optional. Defaults are defined in `secondbrain.config.settings.Settings`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `SB_DATABASE_PATH` | `data/secondbrain.db` | SQLite database file |
| `SB_DOCUMENT_STORAGE_PATH` | `data/documents` | Directory for original uploads |
| `SB_MAX_UPLOAD_SIZE_BYTES` | `10485760` | Maximum upload size (10 MiB) |

The application creates the database parent and document directory as needed. SQLite is embedded; no database server is required. Stored data remains on disk across application restarts.

## Document API

Supported extensions are `.pdf`, `.md`, and `.txt`. Markdown and text are decoded as UTF-8. PDF extraction uses `pypdf`; encrypted, malformed, empty, or textless PDFs are rejected. OCR is not implemented.

Upload a file as multipart form field `file`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -F 'file=@notes.md;type=text/markdown'
```

The response contains metadata only, for example:

```json
{
  "id": "0e5f...",
  "filename": "notes.md",
  "content_type": "text/markdown",
  "size_bytes": 123,
  "created_at": "2026-10-18T10:30:00+00:00",
  "extracted_text_length": 117
}
```

List stored documents (newest first):

```bash
curl http://127.0.0.1:8000/api/v1/documents
```

The response is `{"documents": []}` when there are no documents. Listing returns metadata and text length, not extracted text or file bytes.

Retrieve one document using its 32-character ID from an upload or list response:

```bash
curl http://127.0.0.1:8000/api/v1/documents/0e5f1234567890abcdef1234567890ab
```

The detail response includes the metadata and extracted text:

```json
{
  "id": "0e5f1234567890abcdef1234567890ab",
  "filename": "notes.md",
  "content_type": "text/markdown",
  "size_bytes": 123,
  "created_at": "2026-10-18T10:30:00+00:00",
  "extracted_text_length": 117,
  "extracted_text": "# Notes\\nDocument text"
}
```

A well-formed ID for a missing document returns `404`; an ID that is not 32
hexadecimal characters returns `422`.

## Text chunking

`secondbrain.chunking.chunk_text` provides deterministic character-based
chunks for extracted text. Document ingestion uses it with defaults of 1000
characters per chunk and 150 characters of overlap; callers of the utility can
pass `chunk_size` and `overlap` explicitly. Chunks are saved to SQLite with a
source document ID and zero-based index, preserving their order. They are not
included in upload or retrieval responses.

```python
from secondbrain.chunking import chunk_text

chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
# ["abcdef", "efghij"]
```

Each chunk starts where the prior chunk ends minus the overlap, preserving
the original order and repeated boundary text. Character windows do not
respect words, sentences, or semantic boundaries; Unicode is measured in
Python string characters rather than bytes or model tokens. Whitespace-only
input returns no chunks. Ingestion currently rejects documents whose extracted
text is entirely whitespace.

## Embedding experiment

An embedding is a numerical vector representing text, useful for comparing
meaning by vector similarity. This milestone provides an optional local
provider using Qdrant FastEmbed and `BAAI/bge-small-en-v1.5`, a compact,
384-dimensional English model. FastEmbed uses a quantized ONNX model rather
than requiring the full PyTorch stack. The provider is separate from document
ingestion, and vectors are not persisted.

Install the optional dependency from the repository root:

```bash
python -m pip install -e ".[embeddings]"
```

The first model initialization downloads approximately 67 MB of model files
and needs network access; FastEmbed caches those files for later runs. Create
one provider and reuse it so the model is not initialized for each input:

```python
from secondbrain.application.embeddings import cosine_similarity
from secondbrain.infrastructure.fastembed_provider import FastEmbedProvider

embedder = FastEmbedProvider()
vectors = embedder.embed_many(["A note about gardening", "Growing tomatoes"])
print(len(vectors[0]))  # 384
print(cosine_similarity(vectors[0], vectors[1]))
```

The checked-in tests use a fake FastEmbed-compatible model and require no
download. To run a real-model smoke test after installing the extra, run the
example above in Python. The selected model is English-only and has a 512-token
input limit; longer text may be truncated by the model tokenizer. Model
inference speed depends on CPU and available memory. Embeddings are not stored,
and no semantic search or retrieval workflow exists yet.

## Quality checks

Run these from the repository root (or prefix each command with `./venv/bin/` if your shell does not expose the virtual environment commands):

```bash
pytest
ruff check .
ruff format --check .
mypy src
```

## Project docs

- [Requirements](docs/requirements.md)
- [Initial architecture](docs/architecture.md)

## Current limitations

This slice uses local filesystem storage and SQLite and is intended for local development. It has no OCR, RAG, semantic search, authentication, or cloud storage. The optional embedding experiment computes vectors locally but does not persist them. Filesystem writes and SQLite transactions cannot form one atomic transaction; the application removes the stored file if the database insert fails, but a process crash between the file write and database insert can leave an orphan file. The upload file limit is 10 MiB by default. An ASGI receive guard caps the whole multipart request at the file limit plus 64 KiB for boundaries, headers, and multipart overhead, before FastAPI's parser can spool the over-limit body. A single ASGI message is allocated by the server before middleware can count it, but the middleware does not buffer or replay the request body.
