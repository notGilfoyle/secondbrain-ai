# SecondBrain AI

A personal knowledge management application. The current vertical slice accepts PDF, Markdown, and plain-text files, extracts their text, stores metadata and text in SQLite, retains the original files locally, and lists ingested documents.

## Local setup

Requires Python 3.11 through 3.14.

```bash
python3 -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
uvicorn secondbrain.main:app --reload --app-dir src --no-access-log
```

Uvicorn access logging is disabled in this local launch command because it is a
separate logging path that can include raw request paths and query strings.
The application's structured completion event does not sanitize Uvicorn logs;
if access logging is enabled, configure and review it separately. Keep FastAPI
debug mode disabled outside local development.

The API is available at `http://127.0.0.1:8000`; the health endpoint is `GET /health`, document endpoints are under `/api/v1/documents`, and the OpenAPI UI is `/docs`.

Configuration uses the `SB_` environment-variable prefix. Local `.env` values are optional. Defaults are defined in `secondbrain.config.settings.Settings`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `SB_DATABASE_PATH` | `data/secondbrain.db` | SQLite database file |
| `SB_DOCUMENT_STORAGE_PATH` | `data/documents` | Directory for original uploads |
| `SB_MAX_UPLOAD_SIZE_BYTES` | `10485760` | Maximum upload size (10 MiB) |
| `SB_COHERE_API_KEY` | unset | Optional Cohere API credential; required to construct the Cohere provider |
| `SB_COHERE_MODEL` | unset | Optional Cohere chat model name; required to construct the Cohere provider |
| `SB_COHERE_TIMEOUT_SECONDS` | `30.0` | Cohere HTTP request timeout |

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
there is no vector index or HTTP retrieval endpoint. Missing vectors are not
candidates. Malformed vectors, non-finite
values, and dimension mismatches raise errors instead of being skipped.

### RAG answer generation

`RAGService` combines semantic retrieval with an application-supplied text
generation provider. It sends the question and retrieved context to the
provider, instructing it to use only that context, acknowledge gaps, and treat
document text as untrusted data. The context is JSON-encoded in the user
message. Citations are constructed by the service from retrieval results and
include document ID, chunk ID, filename, and chunk index; model-generated
citation references are not used. Duplicate document/chunk results are
included once.

RAG is invoked by application code; there is no HTTP endpoint. The provider
contract accepts `generate(system_instruction, user_message) -> str`. For
example, with an existing `SemanticRetrievalService` and a configured provider:

```python
from secondbrain.application.rag import RAGRequest, RAGService

rag = RAGService(retriever, text_generation_provider)
response = rag.answer(RAGRequest("How does Python handle asynchronous I/O?"))
print(response.answer)
for citation in response.citations:
    print(citation.filename, citation.document_id, citation.chunk_id)
```

When retrieval finds no sources, the service returns a fixed no-results answer
without calling the generator. Prompt instructions can reduce unsupported
answers and prompt injection, but they cannot guarantee that an LLM follows
those instructions. The answer is not independently verified against its
citations.

#### Cohere provider

`CohereTextGenerationProvider` implements the RAG text-generation contract
using Cohere's non-streaming v2 Chat API over HTTPX. HTTPX is a runtime
dependency. No Cohere SDK is used. Configuration is optional; copy `.env.example`
to `.env`, then set `SB_COHERE_API_KEY` and `SB_COHERE_MODEL` locally. Choose a
model currently enabled for your Cohere account. The default request timeout is
30 seconds and can be changed with `SB_COHERE_TIMEOUT_SECONDS`. The key is
stored as a secret setting and is never included in provider error messages.

Invoke the existing RAG orchestration by injecting the concrete provider:

```python
from pathlib import Path

from secondbrain.application.rag import RAGRequest, RAGService
from secondbrain.application.retrieval import SemanticRetrievalService
from secondbrain.config.settings import get_settings
from secondbrain.infrastructure.cohere_text_generation_provider import (
    CohereTextGenerationProvider,
)
from secondbrain.infrastructure.fastembed_provider import FastEmbedProvider
from secondbrain.infrastructure.sqlite_chunk_embedding_repository import (
    SQLiteChunkEmbeddingRepository,
)

settings = get_settings()
embedding_provider = FastEmbedProvider()
repository = SQLiteChunkEmbeddingRepository(settings.database_path)
retriever = SemanticRetrievalService(embedding_provider, repository)
with CohereTextGenerationProvider(settings) as text_provider:
    rag = RAGService(retriever, text_provider)
    response = rag.answer(RAGRequest("What do my notes say about asyncio?"))
    print(response.answer)
    print(response.citations)
```

The database must already contain documents and embeddings for the selected
embedding model. Normal tests use HTTPX mock transports and make no network
requests: run `pytest tests/test_cohere_text_generation_provider.py`.

For a separate real-model smoke test, install the embedding extra if needed,
configure `.env` with a valid `SB_COHERE_API_KEY` and `SB_COHERE_MODEL`, and
ensure the configured SQLite database contains at least one document chunk
embedded using the `FastEmbedProvider` model. Then run the invocation above
from the repository root with the project's Python environment. It sends the
question and retrieved document text to Cohere. Cohere chat usage is billed by
input and output tokens for production keys; larger retrieved contexts raise
input usage and cost. Check Cohere's [current pricing information](https://docs.cohere.com/docs/how-does-cohere-pricing-work)
and your account's model access before running the smoke test. No credentials
belong in source control or shared documentation.

## HTTP request correlation

Each HTTP request receives a server-generated UUID in the `X-Request-ID`
response header and a JSON `http.request.completed` log event containing its
ID, method, matched route template, response status, and duration, including
unhandled 500 responses. The 500 response keeps Starlette's generic body and
uses the ID stored in request state. A client `X-Request-ID` is recorded
separately only when it passes a bounded ASCII allowlist; it never replaces the
server ID. Request IDs are correlation labels, not authentication or
authorization signals. Events omit concrete paths, query strings, arbitrary
headers, request/response bodies, document text, prompts, answers, credentials,
provider responses, and exception messages. Middleware ordering covers early
upload-size rejections without reading the body twice.

Unexpected document-route failures emit allowlisted operation and failure
category events without exception messages or tracebacks. This safe structured
logging contract applies only to events emitted by the application helper.
Uvicorn access and unhandled-exception logs are separate paths: access logs may
include raw paths and query strings, and Uvicorn error logs may include
tracebacks. The local launch command disables access logs with `--no-access-log`;
the application's completion event does not sanitize Uvicorn logs. Debug mode
is intended only for local development and should be disabled elsewhere.

## Offline retrieval evaluation

The version-controlled fixture in `evaluation/dataset.json` contains eight
synthetic source chunks and ten questions, including two questions with no
supporting evidence. It also records expected answer facts and stable chunk
IDs. Fixed vectors exercise the existing `SemanticRetrievalService` without
model downloads, credentials, or network access. Run the deterministic report
from the repository root:

```bash
venv/bin/python scripts/evaluate_retrieval.py --k 3
```

Recall@k is the fraction of expected supporting chunks present in the first
`k` results, averaged over answerable questions. MRR is the average reciprocal
rank of the first relevant chunk. Questions with no expected supporting chunk
are excluded from both metrics; the report counts answerable questions with no
relevant result in the top k. Unanswerable questions are also excluded from
retrieval false-positive evaluation: this report does not measure how often
they receive irrelevant candidates. The checked-in baseline report is in
[`docs/evaluation-report.md`](docs/evaluation-report.md). These metrics assess
retrieval only. They do not score generated answer quality, abstention, or
whether a Cohere response is factually grounded. The controlled vectors make
the ranking repeatable but do not represent model quality or a production
corpus.

### Offline answer-quality evaluation

Run `venv/bin/python scripts/evaluate_answers.py` for the separate synthetic
answer evaluation. It drives the existing `RAGService` with controlled context
and scripted outputs, checks lexical expected-fact coverage, source citation
tuples, and abstention phrase matches. Evidence support is represented by
checked-in judgment annotations; reviewer provenance is not recorded. No network, Cohere credentials, or model downloads
are used. Scripted results validate evaluator behavior and do not measure real
LLM quality. See [`docs/answer-evaluation-report.md`](docs/answer-evaluation-report.md)
for definitions, limitations, and the optional format for reviewed real outputs.

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

This slice uses local filesystem storage and SQLite and is intended for local development. It has no OCR, authentication, cloud storage, vector index, or HTTP retrieval/answer endpoint. The optional Cohere provider requires account credentials and can incur token-based production charges. Exact semantic retrieval scans stored vectors and suits only a small local corpus. Prompt-based grounding cannot guarantee factual or policy compliance from a model. Filesystem writes and SQLite transactions cannot form one atomic transaction; the application removes the stored file if the database insert fails, but a process crash between the file write and database insert can leave an orphan file. The upload file limit is 10 MiB by default. An ASGI receive guard caps the whole multipart request at the file limit plus 64 KiB for boundaries, headers, and multipart overhead, before FastAPI's parser can spool the over-limit body. A single ASGI message is allocated by the server before middleware can count it, but the middleware does not buffer or replay the request body.
