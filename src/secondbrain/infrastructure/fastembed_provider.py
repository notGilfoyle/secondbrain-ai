"""Local text embeddings backed by Qdrant FastEmbed."""

from collections.abc import Sequence
from importlib import import_module
from typing import Any, cast


class FastEmbedProvider:
    """Generate local vectors using one lazily downloaded FastEmbed model.

    Construct one provider and reuse it for multiple calls. Model weights are
    downloaded and cached by FastEmbed the first time the model is initialized.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            fastembed = import_module("fastembed")
        except ModuleNotFoundError as exc:
            if exc.name == "fastembed":
                raise RuntimeError(
                    "Install the optional embedding dependencies with "
                    "'python -m pip install -e \".[embeddings]\"'"
                ) from exc
            raise
        model_class = cast(Any, fastembed).TextEmbedding
        self._model: Any = model_class(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        """Embed one non-empty string."""
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of non-empty strings in one model call."""
        if any(not text.strip() for text in texts):
            raise ValueError("text inputs must not be empty or whitespace-only")
        if not texts:
            return []

        vectors = list(self._model.embed(list(texts)))
        if len(vectors) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        return [cast(list[float], vector.tolist()) for vector in vectors]
