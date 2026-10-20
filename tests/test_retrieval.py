"""Offline tests for semantic retrieval and SQLite candidate loading."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from secondbrain.application.documents import DocumentService
from secondbrain.application.embeddings import ChunkEmbedding
from secondbrain.application.retrieval import SemanticRetrievalService
from secondbrain.infrastructure.document_parser import DocumentTextExtractor
from secondbrain.infrastructure.local_file_storage import LocalFileStorage
from secondbrain.infrastructure.sqlite_chunk_embedding_repository import (
    SQLiteChunkEmbeddingRepository,
)
from secondbrain.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


class FakeQueryProvider:
    """Fake provider returning one deterministic query vector."""

    def __init__(self, vector: list[float], model_name: str = "fake/model-v1") -> None:
        self.model_name = model_name
        self.vector = vector
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


def make_document(database_path: Path, storage_path: Path, text: str) -> None:
    service = DocumentService(
        DocumentTextExtractor(),
        LocalFileStorage(storage_path),
        SQLiteDocumentRepository(database_path),
        max_upload_size_bytes=len(text.encode("utf-8")) + 1,
    )
    service.ingest("source.txt", "text/plain", text.encode("utf-8"))


def seed_candidates(
    database_path: Path,
    storage_path: Path,
    vectors_by_text: dict[str, tuple[float, ...]],
    model_name: str = "fake/model-v1",
) -> SQLiteChunkEmbeddingRepository:
    repository = SQLiteChunkEmbeddingRepository(database_path)
    for text in vectors_by_text:
        make_document(database_path, storage_path, text)
    chunks = repository.list_chunks_without_embeddings(model_name)
    repository.save_embeddings(
        [
            ChunkEmbedding(
                chunk_id=chunk.id,
                model_name=model_name,
                vector=vectors_by_text[chunk.text],
                created_at=datetime.now(UTC),
            )
            for chunk in chunks
        ]
    )
    return repository


def test_relevant_chunks_rank_first_with_metadata_and_top_k(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    repository = seed_candidates(
        database_path,
        tmp_path / "files",
        {
            "relevant programming note": (0.9, 0.435889894),
            "somewhat related note": (0.8, 0.6),
            "unrelated travel note": (-1.0, 0.0),
        },
    )
    provider = FakeQueryProvider([1.0, 0.0])
    service = SemanticRetrievalService(provider, repository, top_k=2)

    results = service.retrieve("programming query")

    assert [result.text for result in results] == [
        "relevant programming note",
        "somewhat related note",
    ]
    assert [result.similarity for result in results] == sorted(
        (result.similarity for result in results), reverse=True
    )
    assert results[0].chunk_id
    assert results[0].document_id == results[0].document.id
    assert results[0].document.filename == "source.txt"
    assert results[0].chunk_index == 0
    assert provider.calls == ["programming query"]


def test_per_call_top_k_and_model_identity_filtering(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    repository = seed_candidates(
        database_path,
        tmp_path / "files",
        {"selected model result": (1.0, 0.0)},
    )
    chunk = repository.list_chunks_without_embeddings("other/model")
    assert len(chunk) == 1
    repository.save_embeddings(
        [
            ChunkEmbedding(
                chunk_id=chunk[0].id,
                model_name="other/model",
                vector=(100.0, 0.0),
                created_at=datetime.now(UTC),
            )
        ]
    )

    results = SemanticRetrievalService(
        FakeQueryProvider([1.0, 0.0]), repository, top_k=10
    ).retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].text == "selected model result"


def test_empty_corpus_returns_no_results(tmp_path: Path) -> None:
    database_path = tmp_path / "empty.db"
    repository = SQLiteChunkEmbeddingRepository(database_path)
    provider = FakeQueryProvider([1.0, 0.0])

    assert SemanticRetrievalService(provider, repository).retrieve("query") == []
    assert provider.calls == ["query"]


def test_zero_query_vector_is_rejected_for_empty_corpus(tmp_path: Path) -> None:
    repository = SQLiteChunkEmbeddingRepository(tmp_path / "empty.db")
    service = SemanticRetrievalService(FakeQueryProvider([0.0, 0.0]), repository)

    with pytest.raises(ValueError, match="zero-magnitude"):
        service.retrieve("query")


def test_zero_query_vector_is_rejected_for_nonempty_corpus(tmp_path: Path) -> None:
    repository = seed_candidates(
        tmp_path / "documents.db",
        tmp_path / "files",
        {"stored text": (1.0, 0.0)},
    )
    service = SemanticRetrievalService(FakeQueryProvider([0.0, 0.0]), repository)

    with pytest.raises(ValueError, match="zero-magnitude"):
        service.retrieve("query")


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_empty_query_is_rejected(tmp_path: Path, query: str) -> None:
    service = SemanticRetrievalService(
        FakeQueryProvider([1.0, 0.0]),
        SQLiteChunkEmbeddingRepository(tmp_path / "empty.db"),
    )
    with pytest.raises(ValueError, match="query must not be empty"):
        service.retrieve(query)


@pytest.mark.parametrize(
    "top_k", [0, -1, 1.5, "2", float("nan"), float("inf"), True, False]
)
def test_invalid_top_k_is_rejected(tmp_path: Path, top_k: int) -> None:
    repository = SQLiteChunkEmbeddingRepository(tmp_path / "empty.db")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        SemanticRetrievalService(FakeQueryProvider([1.0, 0.0]), repository, top_k)
    service = SemanticRetrievalService(FakeQueryProvider([1.0, 0.0]), repository)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        service.retrieve("query", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, 5, 100])
def test_positive_integer_top_k_is_accepted(tmp_path: Path, top_k: int) -> None:
    repository = SQLiteChunkEmbeddingRepository(tmp_path / "empty.db")
    provider = FakeQueryProvider([1.0, 0.0])

    service = SemanticRetrievalService(provider, repository, top_k=top_k)
    assert service.retrieve("query") == []
    assert service.retrieve("query", top_k=top_k) == []


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ("not-json", "malformed"),
        ('{"value": 1}', "malformed"),
        ('["1", 1.0]', "malformed"),
        ("[1.0]", "invalid dimension"),
        ("[NaN, 1.0]", "finite"),
        ("[Infinity, 1.0]", "finite"),
    ],
)
def test_malformed_or_invalid_stored_vectors_fail_loudly(
    tmp_path: Path, serialized: str, message: str
) -> None:
    database_path = tmp_path / "documents.db"
    repository = seed_candidates(
        database_path,
        tmp_path / "files",
        {"stored text": (1.0, 0.0)},
    )
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                "UPDATE chunk_embeddings SET vector_json = ?", (serialized,)
            )

    with pytest.raises(ValueError, match=message):
        SemanticRetrievalService(FakeQueryProvider([1.0, 0.0]), repository).retrieve(
            "query"
        )


def test_query_and_stored_vector_dimensions_must_match(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    repository = seed_candidates(
        database_path,
        tmp_path / "files",
        {"stored text": (1.0, 0.0)},
    )
    service = SemanticRetrievalService(FakeQueryProvider([1.0, 0.0, 0.0]), repository)

    with pytest.raises(ValueError, match="dimensions must match"):
        service.retrieve("query")


@pytest.mark.parametrize("vector", [[float("nan"), 1.0], [float("inf"), 1.0], []])
def test_invalid_query_vectors_are_rejected(
    tmp_path: Path, vector: list[float]
) -> None:
    repository = SQLiteChunkEmbeddingRepository(tmp_path / "empty.db")
    service = SemanticRetrievalService(FakeQueryProvider(vector), repository)

    with pytest.raises(ValueError):
        service.retrieve("query")
