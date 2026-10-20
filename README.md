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
ingestion.

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
inference speed depends on CPU and available memory.

### Persisting chunk embeddings

Embedding generation is an explicit application use case; ingestion does not
invoke it. After documents have been ingested, run this from the repository
root (with the embedding extra installed) to embed chunks missing vectors for
the selected model:

```python
from pathlib import Path

from secondbrain.application.embedding_generation import EmbeddingGenerationService
from secondbrain.infrastructure.fastembed_provider import FastEmbedProvider
from secondbrain.infrastructure.sqlite_chunk_embedding_repository import (
    SQLiteChunkEmbeddingRepository,
)

provider = FastEmbedProvider()
repository = SQLiteChunkEmbeddingRepository(Path("data/secondbrain.db"))
generation = EmbeddingGenerationService(provider, repository, batch_size=32)
print(generation.embed_missing_chunks())
```

Vectors are stored in SQLite's `chunk_embeddings` table as JSON arrays of
floating-point numbers, which are simple to inspect and adequate for this local
prototype. JSON is larger than packed binary representations and is not
optimized for similarity search. Rows reference chunks with a foreign key and
are unique by `(chunk_id, model_name)`, allowing another model's vector for the
same chunk. `embedding_models` records the model name and its fixed vector
dimension. The model name is the provider identity; when changing a model
revision, use a distinct identity to preserve the distinction.

The use case embeds in batches, holds no SQLite transaction during model
inference, and writes the complete set of generated vectors in one transaction.
Inference or insertion failure leaves no newly persisted vectors from that
invocation. Retrying skips chunks that already have a vector for that model.
All generated vectors are held in memory until persistence, so very large
corpora may need a later bounded, resumable batch workflow.

### Semantic retrieval experiment

Once chunks have stored embeddings, `SemanticRetrievalService` embeds a query
with the selected provider, loads candidates persisted for that same model
identity, and ranks chunks using the existing cosine similarity function. It
returns chunk text and IDs, source document metadata, and a similarity score.
This is exact top-k vector comparison, not an LLM-generated answer: it returns
matching source chunks and does not synthesize or explain a response.

```python
from pathlib import Path

from secondbrain.application.retrieval import SemanticRetrievalService
from secondbrain.infrastructure.fastembed_provider import FastEmbedProvider
from secondbrain.infrastructure.sqlite_chunk_embedding_repository import (
    SQLiteChunkEmbeddingRepository,
)

provider = FastEmbedProvider()
repository = SQLiteChunkEmbeddingRepository(Path("data/secondbrain.db"))
retrieval = SemanticRetrievalService(provider, repository, top_k=5)
for match in retrieval.retrieve("How do I configure asynchronous Python?"):
    print(match.similarity, match.document.filename, match.text)
```

The repository joins vectors, chunks, and document metadata in one query. The
service compares every stored candidate for that model in memory and sorts by
descending cosine similarity, so query time grows linearly with the number and
size of stored vectors. This exact scan is intended for a small local corpus;
there is no vector index, HTTP retrieval endpoint, or answer-generation
pipeline yet. Missing vectors are not candidates. Malformed vectors, non-finite
values, and dimension mismatches raise errors instead of being skipped.

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

This slice uses local filesystem storage and SQLite and is intended for local development. It has no OCR, LLM answer generation, authentication, cloud storage, vector index, or HTTP retrieval endpoint. Exact semantic retrieval scans stored vectors and suits only a small local corpus. Filesystem writes and SQLite transactions cannot form one atomic transaction; the application removes the stored file if the database insert fails, but a process crash between the file write and database insert can leave an orphan file. The upload file limit is 10 MiB by default. An ASGI receive guard caps the whole multipart request at the file limit plus 64 KiB for boundaries, headers, and multipart overhead, before FastAPI's parser can spool the over-limit body. A single ASGI message is allocated by the server before middleware can count it, but the middleware does not buffer or replay the request body.
