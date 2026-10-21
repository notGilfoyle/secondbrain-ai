#!/usr/bin/env python3
"""Run the checked-in deterministic retrieval evaluation fixture."""

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from secondbrain.application.documents import DocumentMetadata
from secondbrain.application.retrieval import (
    EmbeddedChunk,
    SemanticRetrievalService,
)
from secondbrain.evaluation import EvaluationQuery, evaluate_retrieval

EXPECTED_VECTOR_DIMENSIONS = 8


class FixtureEmbeddingProvider:
    """Return the fixed query vector recorded in the synthetic dataset."""

    model_name = "offline-evaluation-v1"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> list[float]:
        try:
            return self._vectors[text]
        except KeyError as exc:
            raise ValueError("query is not part of the evaluation dataset") from exc

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FixtureRetrievalRepository:
    """Expose synthetic chunks through the production retrieval contract."""

    def __init__(self, chunks: list[EmbeddedChunk]) -> None:
        self._chunks = chunks

    def list_candidates(self, model_name: str) -> list[EmbeddedChunk]:
        return [chunk for chunk in self._chunks if chunk.model_name == model_name]


def _load_fixture() -> dict[str, Any]:
    fixture_path = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def validate_fixture(fixture: dict[str, Any]) -> None:
    """Reject inconsistent IDs and vectors before evaluating the fixture."""
    chunks = fixture.get("chunks")
    queries = fixture.get("queries")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("evaluation dataset 'chunks' must be a non-empty list")
    if not isinstance(queries, list) or not queries:
        raise ValueError("evaluation dataset 'queries' must be a non-empty list")

    chunk_ids: list[str] = []
    all_vectors: list[tuple[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"evaluation chunk at index {index} must be an object")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"evaluation chunk at index {index} needs a chunk_id")
        chunk_ids.append(chunk_id)
        all_vectors.append((f"chunk {chunk_id!r}", chunk.get("vector")))
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("evaluation dataset contains duplicate chunk IDs")
    available_chunk_ids = set(chunk_ids)

    query_ids: list[str] = []
    questions: list[str] = []
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"evaluation query at index {index} must be an object")
        query_id = query.get("query_id")
        question = query.get("question")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"evaluation query at index {index} needs a query_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"evaluation query {query_id!r} needs a question")
        query_ids.append(query_id)
        questions.append(question)
        all_vectors.append((f"query {query_id!r}", query.get("query_vector")))

        expected_ids = query.get("expected_chunk_ids")
        if not isinstance(expected_ids, list) or any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in expected_ids
        ):
            raise ValueError(
                f"query {query_id!r} expected_chunk_ids must be a list of IDs"
            )
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError(
                f"query {query_id!r} contains duplicate expected chunk IDs"
            )
        missing_ids = set(expected_ids) - available_chunk_ids
        if missing_ids:
            raise ValueError(
                f"query {query_id!r} references unknown chunk IDs: "
                f"{', '.join(sorted(missing_ids))}"
            )
        facts = query.get("expected_answer_facts")
        if not isinstance(facts, list) or any(
            not isinstance(fact, str) or not fact.strip() for fact in facts
        ):
            raise ValueError(
                f"query {query_id!r} expected_answer_facts must be a list of text"
            )

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluation dataset contains duplicate query IDs")
    if len(questions) != len(set(questions)):
        raise ValueError("evaluation dataset questions must be unique")

    for label, vector in all_vectors:
        if not isinstance(vector, list) or len(vector) != EXPECTED_VECTOR_DIMENSIONS:
            raise ValueError(
                f"{label} vector must have {EXPECTED_VECTOR_DIMENSIONS} dimensions"
            )
        try:
            values_are_finite = all(
                not isinstance(value, bool)
                and isinstance(value, int | float)
                and isfinite(float(value))
                for value in vector
            )
        except (OverflowError, ValueError):
            values_are_finite = False
        if not values_are_finite:
            raise ValueError(f"{label} vector values must be finite numbers")
        if all(value == 0 for value in vector):
            raise ValueError(f"{label} vector must have non-zero magnitude")


def run_evaluation(k: int) -> None:
    fixture = _load_fixture()
    validate_fixture(fixture)
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    chunks = [
        EmbeddedChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            document=DocumentMetadata(
                id=chunk["document_id"],
                filename=chunk["filename"],
                content_type="text/plain",
                storage_path=f"{chunk['document_id']}.txt",
                size_bytes=len(chunk["text"].encode("utf-8")),
                created_at=timestamp,
                extracted_text_length=len(chunk["text"]),
            ),
            model_name=FixtureEmbeddingProvider.model_name,
            vector=tuple(chunk["vector"]),
        )
        for chunk in fixture["chunks"]
    ]
    queries_data = fixture["queries"]
    provider = FixtureEmbeddingProvider(
        {item["question"]: item["query_vector"] for item in queries_data}
    )
    service = SemanticRetrievalService(
        provider,
        FixtureRetrievalRepository(chunks),
        top_k=k,
    )
    queries = [
        EvaluationQuery(
            query_id=item["query_id"],
            question=item["question"],
            relevant_chunk_ids=tuple(item["expected_chunk_ids"]),
        )
        for item in queries_data
    ]
    report = evaluate_retrieval(service, queries, k=k)
    print("Offline semantic retrieval evaluation")
    print(f"Dataset queries: {report.query_count}")
    print(f"Answerable queries scored: {report.answerable_query_count}")
    print(f"Recall@{k}: {report.recall_at_k:.3f}")
    print(f"MRR: {report.mean_reciprocal_rank:.3f}")
    print(
        f"Answerable queries with no relevant chunk in top {k}: "
        f"{report.no_relevant_chunk_queries}"
    )
    print("Unanswerable cases are excluded; this does not score abstention behavior.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="ranking cutoff (default: 3)")
    args = parser.parse_args()
    run_evaluation(args.k)


if __name__ == "__main__":
    main()
