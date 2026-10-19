"""Application-level contracts and operations for text embeddings."""

from collections.abc import Sequence
from math import fsum, hypot, isfinite
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Produce vectors for one text or a batch of texts."""

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two finite, non-zero vectors."""
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    if not left:
        raise ValueError("embeddings must not be empty")

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not all(isfinite(value) for value in (*left_values, *right_values)):
        raise ValueError("embedding values must be finite")

    left_norm = hypot(*left_values)
    right_norm = hypot(*right_values)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero-magnitude embeddings have no cosine similarity")

    similarity = fsum(
        (left_value / left_norm) * (right_value / right_norm)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    return max(-1.0, min(1.0, similarity))
