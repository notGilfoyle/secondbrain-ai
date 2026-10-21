"""Tests for deterministic retrieval evaluation metrics."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.evaluate_retrieval import validate_fixture

from secondbrain.application.documents import DocumentMetadata
from secondbrain.application.retrieval import RetrievalResult
from secondbrain.evaluation import EvaluationQuery, evaluate_retrieval


class FixedRankedRetriever:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        chunk_ids = self.rankings[query]
        if top_k is not None:
            chunk_ids = chunk_ids[:top_k]
        return [make_result(chunk_id) for chunk_id in chunk_ids]


def make_result(chunk_id: str) -> RetrievalResult:
    document = DocumentMetadata(
        id="doc-1",
        filename="synthetic.txt",
        content_type="text/plain",
        storage_path="doc-1.txt",
        size_bytes=10,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        extracted_text_length=10,
    )
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        text="Synthetic fact.",
        document=document,
        similarity=1.0,
    )


def test_evaluation_calculates_recall_mrr_and_missed_queries() -> None:
    queries = [
        EvaluationQuery("first", "first question", ("a",)),
        EvaluationQuery("second", "second question", ("b", "c")),
        EvaluationQuery("third", "third question", ("d",)),
        EvaluationQuery("unanswerable", "unanswerable question", ()),
    ]
    report = evaluate_retrieval(
        FixedRankedRetriever(
            {
                "first question": ["x", "a"],
                "second question": ["b", "x", "c"],
                "third question": ["x", "y"],
            }
        ),
        queries,
        k=2,
    )

    assert report.query_count == 4
    assert report.answerable_query_count == 3
    assert report.recall_at_k == 0.5
    assert report.mean_reciprocal_rank == 0.5
    assert report.no_relevant_chunk_queries == 1


def test_duplicate_expected_chunk_ids_are_rejected() -> None:
    query = EvaluationQuery("duplicate", "question", ("a", "a"))

    with pytest.raises(ValueError, match="duplicate relevant chunk IDs"):
        evaluate_retrieval(
            FixedRankedRetriever({"question": ["a"]}),
            [query],
            k=1,
        )


def test_checked_in_fixture_passes_validation() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)

    validate_fixture(fixture)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda fixture: fixture["chunks"].append(fixture["chunks"][0].copy()),
            "duplicate chunk IDs",
        ),
        (
            lambda fixture: fixture["queries"].append(fixture["queries"][0].copy()),
            "duplicate query IDs",
        ),
        (
            lambda fixture: fixture["queries"][0]["expected_chunk_ids"].append(
                "missing-chunk"
            ),
            "references unknown chunk IDs",
        ),
        (
            lambda fixture: fixture["chunks"][0]["vector"].pop(),
            "must have 8 dimensions",
        ),
        (
            lambda fixture: fixture["queries"][0]["query_vector"].__setitem__(
                0, float("nan")
            ),
            "finite numbers",
        ),
    ],
)
def test_invalid_fixture_is_rejected(mutate: object, message: str) -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = deepcopy(json.load(fixture_file))
    mutate(fixture)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        validate_fixture(fixture)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_evaluation_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate_retrieval(FixedRankedRetriever({}), [], k=k)
