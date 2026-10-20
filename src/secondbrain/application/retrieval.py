"""Application use case for exact semantic retrieval over stored chunk vectors."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from secondbrain.application.documents import DocumentMetadata
from secondbrain.application.embeddings import EmbeddingProvider, cosine_similarity


@dataclass(frozen=True)
class EmbeddedChunk:
    """A stored chunk, its owning document metadata, and its model vector."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    document: DocumentMetadata
    model_name: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """A matching chunk and document with its query similarity score."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    document: DocumentMetadata
    similarity: float


class RetrievalRepository(Protocol):
    """Load stored chunk vectors with their chunk and document data."""

    def list_candidates(self, model_name: str) -> list[EmbeddedChunk]:
        """Return candidate vectors for exactly one embedding model identity."""
        ...


class SemanticRetrievalService:
    """Rank stored document chunks by exact cosine similarity to a query."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        repository: RetrievalRepository,
        top_k: int = 5,
    ) -> None:
        self._top_k = self._validate_top_k(top_k)
        if not provider.model_name.strip():
            raise ValueError("embedding provider model_name must not be empty")
        self._provider = provider
        self._repository = repository

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return up to ``top_k`` chunks ranked by descending cosine similarity."""
        result_limit = self._top_k if top_k is None else self._validate_top_k(top_k)
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        query_vector = self._validate_vector(self._provider.embed(query))
        # Reuse the shared cosine validator to reject zero magnitude even when
        # there are no candidates to compare against.
        cosine_similarity(query_vector, query_vector)
        candidates = self._repository.list_candidates(self._provider.model_name)
        scored: list[RetrievalResult] = []
        for candidate in candidates:
            if candidate.model_name != self._provider.model_name:
                raise ValueError("repository returned a different embedding model")
            vector = self._validate_vector(candidate.vector)
            similarity = cosine_similarity(query_vector, vector)
            scored.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    chunk_index=candidate.chunk_index,
                    text=candidate.text,
                    document=candidate.document,
                    similarity=similarity,
                )
            )

        scored.sort(key=lambda result: result.similarity, reverse=True)
        return scored[:result_limit]

    @staticmethod
    def _validate_top_k(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("top_k must be a positive integer")
        return value

    @staticmethod
    def _validate_vector(vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        if not vector:
            raise ValueError("embedding vectors must not be empty")
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding vector values must be numeric") from exc
        if not all(isfinite(value) for value in values):
            raise ValueError("embedding vector values must be finite")
        return values
