"""Validation and deterministic answer evaluation checks."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from secondbrain.answer_evaluation import (
    evaluate_answer_case,
    retrieved_source_attribution_metrics,
    validate_answer_fixture,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data() -> tuple[dict, list[dict]]:
    source = json.loads((ROOT / "evaluation/dataset.json").read_text(encoding="utf-8"))
    fixture = json.loads(
        (ROOT / "evaluation/answer_dataset.json").read_text(encoding="utf-8")
    )
    return fixture, source["chunks"]


def test_answer_fixture_validates_and_covers_required_scenarios(
    data: tuple[dict, list[dict]],
) -> None:
    fixture, chunks = data
    validate_answer_fixture(fixture, chunks)
    kinds = {case["case_type"] for case in fixture["cases"]}
    assert kinds == {"answerable", "partially_sufficient", "unanswerable"}
    assert any(not case["retrieved_chunk_ids"] for case in fixture["cases"])
    assert any(
        case["retrieved_chunk_ids"] and case["case_type"] == "unanswerable"
        for case in fixture["cases"]
    )
    assert any(case.get("unsupported_claims") for case in fixture["cases"])
    assert any(len(case["retrieved_chunk_ids"]) > 1 for case in fixture["cases"])


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda f: f["cases"].append(deepcopy(f["cases"][0])), "unique"),
        (lambda f: f["cases"][0]["retrieved_chunk_ids"].append("missing"), "unknown"),
        (lambda f: f["cases"][0]["fact_support"].clear(), "annotate every"),
        (
            lambda f: f["cases"][0]["fact_support"]["May 12"].update(
                {"status": "supported", "chunk_ids": []}
            ),
            "needs evidence",
        ),
        (
            lambda f: f["cases"][0]["fact_support"]["May 12"].update(
                {"status": "supported", "chunk_ids": ["unknown"]}
            ),
            "unknown chunk",
        ),
        (
            lambda f: f["cases"][0]["fact_support"]["May 12"].update(
                {"status": "withheld", "chunk_ids": ["chunk-release"]}
            ),
            "cannot cite evidence",
        ),
        (
            lambda f: f["cases"][0]["expected_citation_chunk_ids"].append(
                "chunk-budget"
            ),
            "unknown",
        ),
    ],
)
def test_invalid_fixture_is_rejected(
    data: tuple[dict, list[dict]], mutation: object, message: str
) -> None:
    fixture, chunks = data
    mutation(fixture)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        validate_answer_fixture(fixture, chunks)


def test_controlled_rag_case_checks_facts_and_exact_source_citations(
    data: tuple[dict, list[dict]],
) -> None:
    fixture, source = data
    validate_answer_fixture(fixture, source)
    chunks = {chunk["chunk_id"]: chunk for chunk in source}
    result = evaluate_answer_case(
        fixture["cases"][0], chunks, fixture["abstention_phrases"]
    )
    assert (result.matched_fact_count, result.expected_fact_count) == (1, 1)
    assert result.retrieved_source_attribution_precision == 1.0
    assert result.retrieved_source_attribution_recall == 1.0


def test_fixture_rejects_duplicate_retrieved_chunk_ids(
    data: tuple[dict, list[dict]],
) -> None:
    fixture, source = data
    case = deepcopy(fixture["cases"][0])
    case["retrieved_chunk_ids"] = ["chunk-release", "chunk-release"]
    with pytest.raises(ValueError, match="retrieved_chunk_ids contains duplicates"):
        validate_answer_fixture({"version": 1, "cases": [case]}, source)


def test_retrieved_source_attribution_handles_missing_unexpected_and_empty_sets() -> (
    None
):
    expected = [("doc", "chunk-1", "notes.md", 0)]
    unexpected = [("doc", "chunk-2", "other.md", 1)]
    assert retrieved_source_attribution_metrics(expected, []) == (1.0, 0.0)
    assert retrieved_source_attribution_metrics(expected, unexpected) == (0.0, 0.0)
    assert retrieved_source_attribution_metrics([], []) == (1.0, 1.0)
    assert retrieved_source_attribution_metrics([], unexpected) == (0.0, 1.0)


def test_retrieved_source_attribution_rejects_duplicate_tuples() -> None:
    expected = [("doc", "chunk-1", "notes.md", 0)]
    with pytest.raises(ValueError, match="actual citation tuples must be unique"):
        retrieved_source_attribution_metrics(expected, expected + expected)


@pytest.mark.parametrize(
    "mismatched_tuple",
    [
        ("other-doc", "chunk-1", "notes.md", 0),
        ("doc", "other-chunk", "notes.md", 0),
        ("doc", "chunk-1", "wrong.md", 0),
        ("doc", "chunk-1", "notes.md", 1),
    ],
)
def test_retrieved_source_attribution_rejects_metadata_mismatch(
    mismatched_tuple: tuple[str, str, str, int],
) -> None:
    expected = [("doc", "chunk-1", "notes.md", 0)]
    assert retrieved_source_attribution_metrics(expected, [mismatched_tuple]) == (
        0.0,
        0.0,
    )


def test_declared_unsupported_claim_is_flagged_and_omitted_claim_passes(
    data: tuple[dict, list[dict]],
) -> None:
    fixture, source = data
    validate_answer_fixture(fixture, source)
    chunks = {chunk["chunk_id"]: chunk for chunk in source}
    case = next(
        case
        for case in fixture["cases"]
        if case["case_id"] == "unsupported-plausible-claim"
    )
    valid = evaluate_answer_case(case, chunks, fixture["abstention_phrases"])
    assert valid.unsupported_claim_matches == ()
    bad_case = deepcopy(case)
    bad_case["scripted_answer"] += " The Northstar release is on Thursday at 10:00."
    flagged = evaluate_answer_case(bad_case, chunks, fixture["abstention_phrases"])
    assert flagged.unsupported_claim_matches == tuple(bad_case["unsupported_claims"])


def test_abstention_phrase_match_is_only_a_proxy(data: tuple[dict, list[dict]]) -> None:
    fixture, source = data
    case = next(
        case
        for case in fixture["cases"]
        if case["case_id"] == "abstention-phrase-false-positive"
    )
    chunks = {chunk["chunk_id"]: chunk for chunk in source}
    result = evaluate_answer_case(case, chunks, fixture["abstention_phrases"])
    assert result.abstention_phrase_match
    assert result.unsupported_claim_matches == tuple(case["unsupported_claims"])
