"""Offline tests for embedding validation, batching, and similarity math."""

import sys
from math import isfinite
from types import ModuleType

import pytest

from secondbrain.application.embeddings import EmbeddingProvider, cosine_similarity
from secondbrain.infrastructure.fastembed_provider import FastEmbedProvider


class FakeVector:
    """Small stand-in for FastEmbed's NumPy vector output."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeModel:
    """Deterministic fake that behaves like FastEmbed's batch interface."""

    instances = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []
        FakeModel.instances += 1

    def embed(self, texts: list[str]) -> list[FakeVector]:
        self.calls.append(texts)
        return [FakeVector([float(len(text)), 1.0, 2.0]) for text in texts]


def make_provider(monkeypatch: pytest.MonkeyPatch) -> FastEmbedProvider:
    module = ModuleType("fastembed")
    module.TextEmbedding = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return FastEmbedProvider()


def test_provider_returns_finite_vectors_and_reuses_one_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeModel.instances = 0
    provider = make_provider(monkeypatch)
    vectors = provider.embed_many(["first", "second"])
    one = provider.embed("x")

    assert vectors == [[5.0, 1.0, 2.0], [6.0, 1.0, 2.0]]
    assert one == [1.0, 1.0, 2.0]
    assert len(vectors[0]) == len(vectors[1]) == len(one) == 3
    assert all(isfinite(value) for vector in [*vectors, one] for value in vector)
    assert FakeModel.instances == 1


def test_provider_uses_configured_model_and_batches_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("fastembed")
    module.TextEmbedding = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    provider = FastEmbedProvider("custom/model")
    assert provider._model.model_name == "custom/model"
    assert provider.embed_many(["a", "bb"]) == [
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 2.0],
    ]
    assert provider._model.calls == [["a", "bb"]]


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_provider_rejects_empty_text(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    provider = make_provider(monkeypatch)
    with pytest.raises(ValueError, match="empty"):
        provider.embed(text)


def test_provider_empty_batch_returns_no_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = make_provider(monkeypatch)
    assert provider.embed_many([]) == []


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
    ],
)
def test_cosine_similarity(
    left: list[float], right: list[float], expected: float
) -> None:
    assert cosine_similarity(left, right) == pytest.approx(expected)


def test_cosine_similarity_supports_large_finite_magnitudes() -> None:
    assert cosine_similarity([1e308, 0.0], [1e308, 0.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([], [], "empty"),
        ([1.0], [1.0, 2.0], "dimensions"),
        ([0.0, 0.0], [1.0, 0.0], "zero-magnitude"),
        ([float("nan")], [1.0], "finite"),
        ([float("inf")], [1.0], "finite"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left: list[float], right: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)


def test_fake_provider_satisfies_application_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider: EmbeddingProvider = make_provider(monkeypatch)
    assert len(provider.embed("a")) == len(provider.embed("longer"))
