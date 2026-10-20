"""Use case for generating and persisting embeddings for stored chunks."""

from collections.abc import Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Protocol

from secondbrain.application.documents import DocumentChunk
from secondbrain.application.embeddings import ChunkEmbedding, EmbeddingProvider


class ChunkEmbeddingRepository(Protocol):
    """Load unembedded chunks and persist/retrieve their vectors."""

    def list_chunks_without_embeddings(self, model_name: str) -> list[DocumentChunk]:
        """Return chunks without an embedding for the requested model."""
        ...

    def save_embeddings(self, embeddings: Sequence[ChunkEmbedding]) -> int:
        """Persist vectors atomically and return the number newly inserted."""
        ...

    def list_embeddings(self, model_name: str) -> list[ChunkEmbedding]:
        """Return vectors for the requested model."""
        ...


class EmbeddingGenerationService:
    """Generate vectors for missing chunks and persist them as one batch."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        repository: ChunkEmbeddingRepository,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not provider.model_name.strip():
            raise ValueError("embedding provider model_name must not be empty")
        self._provider = provider
        self._repository = repository
        self._batch_size = batch_size

    def embed_missing_chunks(self) -> int:
        """Embed unprocessed chunks and return the number of vectors stored.

        All inference completes before the repository is asked to persist. A
        provider failure therefore leaves the database untouched for this run.
        """
        model_name = self._provider.model_name
        chunks = self._repository.list_chunks_without_embeddings(model_name)
        if not chunks:
            return 0

        chunk_ids = [chunk.id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("embedding batch contains duplicate chunk IDs")

        records: list[ChunkEmbedding] = []
        expected_dimensions: int | None = None
        created_at = datetime.now(UTC)
        for offset in range(0, len(chunks), self._batch_size):
            batch = chunks[offset : offset + self._batch_size]
            vectors = self._provider.embed_many([chunk.text for chunk in batch])
            if len(vectors) != len(batch):
                raise ValueError(
                    "embedding provider returned an unexpected vector count"
                )

            for chunk, vector in zip(batch, vectors, strict=True):
                values = self._validate_vector(vector)
                if expected_dimensions is None:
                    expected_dimensions = len(values)
                elif len(values) != expected_dimensions:
                    raise ValueError(
                        "embedding vectors must have consistent dimensions"
                    )
                records.append(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        model_name=model_name,
                        vector=values,
                        created_at=created_at,
                    )
                )

        return self._repository.save_embeddings(records)

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
        if not vector:
            raise ValueError("embedding vectors must not be empty")
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding vector values must be numeric") from exc
        if not all(isfinite(value) for value in values):
            raise ValueError("embedding vector values must be finite")
        return values
