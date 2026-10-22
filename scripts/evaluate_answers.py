#!/usr/bin/env python3
"""Run deterministic, offline RAG answer evaluation cases."""

import json
from pathlib import Path
from typing import Any

from secondbrain.answer_evaluation import evaluate_answer_case, validate_answer_fixture


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "evaluation/dataset.json").open(encoding="utf-8") as stream:
        source = json.load(stream)
    with (root / "evaluation/answer_dataset.json").open(encoding="utf-8") as stream:
        fixture: dict[str, Any] = json.load(stream)
    validate_answer_fixture(fixture, source["chunks"])
    chunks = {chunk["chunk_id"]: chunk for chunk in source["chunks"]}
    results = [
        evaluate_answer_case(case, chunks, fixture["abstention_phrases"])
        for case in fixture["cases"]
    ]
    matched = sum(result.matched_fact_count for result in results)
    facts = sum(result.expected_fact_count for result in results)
    matched_supported = sum(result.matched_supported_fact_count for result in results)
    supported_facts = sum(result.supported_fact_count for result in results)
    print("Offline RAG answer evaluation (scripted provider; not model quality)")
    print(f"Cases: {len(results)}")
    fact_rate = matched / facts if facts else 1.0
    print(
        f"Lexical expected-fact exact-match proxy: {matched}/{facts} ({fact_rate:.3f})"
    )
    print("Checked-in judgment annotations (not model judgments):")
    for label in ("supported", "contradicted", "unsupported"):
        count = sum(len(case["judgments"][label]) for case in fixture["cases"])
        print(f"  {label}: {count}")
    supported_fact_rate = (
        matched_supported / supported_facts if supported_facts else 1.0
    )
    print(
        "Supported expected-fact exact-match proxy: "
        f"{matched_supported}/{supported_facts} ({supported_fact_rate:.3f})"
    )
    attribution_precision = sum(
        r.retrieved_source_attribution_precision for r in results
    ) / len(results)
    attribution_recall = sum(
        r.retrieved_source_attribution_recall for r in results
    ) / len(results)
    print(
        f"Retrieved-source attribution precision: {attribution_precision:.3f} (macro)"
    )
    print(f"Retrieved-source attribution recall: {attribution_recall:.3f} (macro)")
    unsupported_matches = sum(len(r.unsupported_claim_matches) for r in results)
    unsupported_claim_count = sum(
        len(case.get("unsupported_claims", [])) for case in fixture["cases"]
    )
    print(
        "Unsupported-claim exact-phrase matches (lexical check): "
        f"{unsupported_matches}/{unsupported_claim_count}"
    )
    abstain = [
        r
        for r, case in zip(results, fixture["cases"], strict=True)
        if case["case_type"] == "unanswerable"
    ]
    hits = sum(result.abstention_phrase_match for result in abstain)
    print(
        "Abstention phrase-match proxy (not verified abstention): "
        f"{hits}/{len(abstain)}"
    )
    print("Retrieval ranking is evaluated separately by scripts/evaluate_retrieval.py.")


if __name__ == "__main__":
    main()
