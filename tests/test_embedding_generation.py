"""Offline tests for embedding generation and SQLite persistence."""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from secondbrain.application.documents import DocumentChunk, DocumentService
from secondbrain.application.embedding_generation import EmbeddingGenerationService
from secondbrain.application.embeddings import ChunkEmbedding
from secondbrain.infrastructure.document_parser import DocumentTextExtractor
from secondbrain.infrastructure.local_file_storage import LocalFileStorage
from secondbrain.infrastructure.sqlite_chunk_embedding_repository import (
    SQLiteChunkEmbeddingRepository,
)
from secondbrain.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


class FakeEmbeddingProvider:
    """Deterministic provider that records batch calls and can fail on demand."""

    def __init__(
        self,
        model_name: str = "fake/model-v1",
        dimensions: int = 2,
        fail_on_call: int | None = None,
        vectors: list[list[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.fail_on_call = fail_on_call
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated model failure")
        if self.vectors is not None:
            start = sum(len(call) for call in self.calls[:-1])
            return self.vectors[start : start + len(texts)]
        return [
            [float(len(text)), float(sum(text.encode("utf-8")) % 31 + 1)]
            + [1.0] * (self.dimensions - 2)
            for text in texts
        ]


def make_document(database_path: Path, storage_path: Path, text: str) -> None:
    service = DocumentService(
        DocumentTextExtractor(),
        LocalFileStorage(storage_path),
        SQLiteDocumentRepository(database_path),
        max_upload_size_bytes=len(text.encode("utf-8")) + 1,
    )
    service.ingest("source.txt", "text/plain", text.encode("utf-8"))


def count_rows(database_path: Path, table_name: str) -> int:
    with closing(sqlite3.connect(database_path)) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def test_embeddings_persist_and_retrieve_mapped_to_chunk_ids(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    text = "a" * 1200 + "b" * 20
    make_document(database_path, tmp_path / "files", text)
    provider = FakeEmbeddingProvider()
    repository = SQLiteChunkEmbeddingRepository(database_path)

    count = EmbeddingGenerationService(provider, repository).embed_missing_chunks()
    persisted = repository.list_embeddings(provider.model_name)

    assert count == 2
    assert len(persisted) == 2
    assert all(entry.model_name == provider.model_name for entry in persisted)
    assert all(entry.dimensions == 2 for entry in persisted)
    with closing(sqlite3.connect(database_path)) as connection:
        chunk_texts = dict(
            connection.execute("SELECT id, text FROM document_chunks").fetchall()
        )
    for entry in persisted:
        chunk_text = chunk_texts[entry.chunk_id]
        expected = [
            float(len(chunk_text)),
            float(sum(chunk_text.encode("utf-8")) % 31 + 1),
        ]
        assert entry.vector == tuple(expected)


def test_chunks_are_inferred_in_batches_and_retries_are_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "x" * 3500)
    provider = FakeEmbeddingProvider()
    repository = SQLiteChunkEmbeddingRepository(database_path)
    service = EmbeddingGenerationService(provider, repository, batch_size=2)

    assert service.embed_missing_chunks() == 4
    assert [len(call) for call in provider.calls] == [2, 2]
    assert service.embed_missing_chunks() == 0
    assert [len(call) for call in provider.calls] == [2, 2]
    assert count_rows(database_path, "chunk_embeddings") == 4


def test_empty_database_returns_zero_without_model_calls(tmp_path: Path) -> None:
    database_path = tmp_path / "empty.db"
    provider = FakeEmbeddingProvider()
    repository = SQLiteChunkEmbeddingRepository(database_path)

    assert EmbeddingGenerationService(provider, repository).embed_missing_chunks() == 0
    assert provider.calls == []


def test_service_rejects_duplicate_chunk_ids_before_inference(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "short source")
    base_repository = SQLiteChunkEmbeddingRepository(database_path)

    class DuplicateChunkRepository(SQLiteChunkEmbeddingRepository):
        def list_chunks_without_embeddings(
            self, model_name: str
        ) -> list[DocumentChunk]:
            chunks = super().list_chunks_without_embeddings(model_name)
            return [*chunks, *chunks]

    repository = DuplicateChunkRepository(database_path)
    provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        EmbeddingGenerationService(provider, repository).embed_missing_chunks()

    assert provider.calls == []
    assert count_rows(database_path, "chunk_embeddings") == 0
    assert count_rows(database_path, "embedding_models") == 0
    assert base_repository.list_embeddings(provider.model_name) == []


def test_repository_rejects_duplicate_chunk_ids_without_partial_writes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "short source")
    repository = SQLiteChunkEmbeddingRepository(database_path)
    chunk = repository.list_chunks_without_embeddings("fake/model-v1")[0]
    now = datetime.now(UTC)
    embeddings = [
        ChunkEmbedding(chunk.id, "fake/model-v1", (1.0, 2.0), now),
        ChunkEmbedding(chunk.id, "fake/model-v1", (3.0, 4.0), now),
    ]

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        repository.save_embeddings(embeddings)

    assert count_rows(database_path, "chunk_embeddings") == 0
    assert count_rows(database_path, "embedding_models") == 0


def test_same_chunk_can_have_vectors_for_different_models(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "short source")
    repository = SQLiteChunkEmbeddingRepository(database_path)
    first = FakeEmbeddingProvider("model-one", dimensions=2)
    second = FakeEmbeddingProvider("model-two", dimensions=3)

    assert EmbeddingGenerationService(first, repository).embed_missing_chunks() == 1
    assert EmbeddingGenerationService(second, repository).embed_missing_chunks() == 1
    assert repository.list_embeddings("model-one")[0].dimensions == 2
    assert repository.list_embeddings("model-two")[0].dimensions == 3
    assert count_rows(database_path, "chunk_embeddings") == 2


def test_model_identity_cannot_change_dimensions(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "short source")
    repository = SQLiteChunkEmbeddingRepository(database_path)
    first = FakeEmbeddingProvider("same-model", dimensions=2)
    changed = FakeEmbeddingProvider("same-model", dimensions=3)
    assert EmbeddingGenerationService(first, repository).embed_missing_chunks() == 1
    make_document(database_path, tmp_path / "files", "new chunk")

    with pytest.raises(ValueError, match="stored model identity"):
        EmbeddingGenerationService(changed, repository).embed_missing_chunks()
    assert len(repository.list_embeddings("same-model")) == 1


@pytest.mark.parametrize(
    "vectors",
    [
        [[]],
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
    ],
)
def test_invalid_vectors_are_rejected_before_persistence(
    tmp_path: Path, vectors: list[list[float]]
) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "text")
    repository = SQLiteChunkEmbeddingRepository(database_path)
    provider = FakeEmbeddingProvider(vectors=vectors)

    with pytest.raises(ValueError):
        EmbeddingGenerationService(provider, repository).embed_missing_chunks()
    assert count_rows(database_path, "chunk_embeddings") == 0
    assert count_rows(database_path, "embedding_models") == 0


def test_inconsistent_dimensions_are_rejected_before_persistence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "x" * 1200)
    repository = SQLiteChunkEmbeddingRepository(database_path)
    provider = FakeEmbeddingProvider(vectors=[[1.0, 2.0], [1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="consistent dimensions"):
        EmbeddingGenerationService(provider, repository).embed_missing_chunks()
    assert count_rows(database_path, "chunk_embeddings") == 0


def test_model_failure_midway_leaves_no_partial_embeddings(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "x" * 3500)
    repository = SQLiteChunkEmbeddingRepository(database_path)
    provider = FakeEmbeddingProvider(fail_on_call=2)
    service = EmbeddingGenerationService(provider, repository, batch_size=2)

    with pytest.raises(RuntimeError, match="simulated model failure"):
        service.embed_missing_chunks()
    assert len(provider.calls) == 2
    assert count_rows(database_path, "chunk_embeddings") == 0
    assert count_rows(database_path, "embedding_models") == 0


def test_missing_chunk_reference_is_rejected_by_foreign_key(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    SQLiteDocumentRepository(database_path).list()
    repository = SQLiteChunkEmbeddingRepository(database_path)
    now = datetime.now(UTC)
    invalid = ChunkEmbedding("missing-chunk", "fake/model-v1", (1.0, 2.0), now)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        repository.save_embeddings([invalid])
    assert count_rows(database_path, "chunk_embeddings") == 0
    assert count_rows(database_path, "embedding_models") == 0


def test_embedding_batch_rolls_back_when_sqlite_insert_fails(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"
    make_document(database_path, tmp_path / "files", "x" * 1200)
    repository = SQLiteChunkEmbeddingRepository(database_path)
    assert repository.list_chunks_without_embeddings("fake/model-v1")
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TRIGGER reject_one_embedding
                BEFORE INSERT ON chunk_embeddings
                WHEN NEW.chunk_id = (
                    SELECT id FROM document_chunks
                    ORDER BY chunk_index LIMIT 1 OFFSET 1
                )
                BEGIN
                    SELECT RAISE(ABORT, 'simulated vector insertion failure');
                END
                """
            )
    provider = FakeEmbeddingProvider()

    with pytest.raises(sqlite3.IntegrityError, match="simulated vector insertion"):
        EmbeddingGenerationService(provider, repository).embed_missing_chunks()

    assert count_rows(database_path, "chunk_embeddings") == 0
    assert count_rows(database_path, "embedding_models") == 0
