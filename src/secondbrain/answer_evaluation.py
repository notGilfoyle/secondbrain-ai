"""Deterministic checks for the offline RAG answer evaluation fixture."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from secondbrain.application.documents import DocumentMetadata
from secondbrain.application.rag import Citation, RAGRequest, RAGService
from secondbrain.application.retrieval import RetrievalResult


@dataclass(frozen=True)
class AnswerCaseResult:
    case_id: str
    expected_fact_count: int
    matched_fact_count: int
    supported_fact_count: int
    matched_supported_fact_count: int
    retrieved_source_attribution_precision: float
    retrieved_source_attribution_recall: float
    unsupported_claim_matches: tuple[str, ...]
    abstention_phrase_match: bool


def retrieved_source_attribution_metrics(
    expected: Sequence[tuple[str, str, str, int]],
    actual: Sequence[tuple[str, str, str, int]],
) -> tuple[float, float]:
    """Compare retrieved-source citation tuples, rejecting duplicate tuples.

    Empty actual citations use precision 1 by convention; recall is 0 when
    citations were expected. Both metrics are 1 when both sets are empty.
    """
    if len(expected) != len(set(expected)):
        raise ValueError("expected citation tuples must be unique")
    if len(actual) != len(set(actual)):
        raise ValueError("actual citation tuples must be unique")
    expected_set, actual_set = set(expected), set(actual)
    true_positives = len(expected_set & actual_set)
    precision = true_positives / len(actual_set) if actual_set else 1.0
    recall = true_positives / len(expected_set) if expected_set else 1.0
    return precision, recall


class _ControlledRetriever:
    def __init__(self, results: list[Any]) -> None:
        self.results = results

    def retrieve(self, query: str) -> list[Any]:
        return self.results


class _ScriptedProvider:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, system_instruction: str, user_message: str) -> str:
        return self.answer


def validate_answer_fixture(
    fixture: Mapping[str, Any], source_chunks: Sequence[Mapping[str, Any]]
) -> None:
    """Validate case IDs, source references, evidence annotations and expectations."""
    chunks = {chunk.get("chunk_id"): chunk for chunk in source_chunks}
    if len(chunks) != len(source_chunks) or None in chunks:
        raise ValueError("source fixture contains duplicate or missing chunk IDs")
    for chunk_id, chunk in chunks.items():
        if (
            not isinstance(chunk.get("document_id"), str)
            or not isinstance(chunk.get("filename"), str)
            or not isinstance(chunk.get("chunk_index"), int)
            or isinstance(chunk.get("chunk_index"), bool)
            or not isinstance(chunk.get("text"), str)
        ):
            raise ValueError(
                f"source chunk {chunk_id!r} has incomplete citation metadata"
            )
    cases = fixture.get("cases")
    if fixture.get("version") != 1 or not isinstance(cases, list) or not cases:
        raise ValueError("answer dataset must have version 1 and non-empty cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("answer case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("answer case IDs must be non-empty and unique")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"case {case_id!r} needs a question")
        if case.get("case_type") not in {
            "answerable",
            "partially_sufficient",
            "unanswerable",
        }:
            raise ValueError(f"case {case_id!r} has invalid case_type")
        for field in (
            "retrieved_chunk_ids",
            "expected_citation_chunk_ids",
            "expected_facts",
        ):
            values = case.get(field)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(
                    f"case {case_id!r} {field} must be a list of non-empty strings"
                )
            if len(values) != len(set(values)):
                raise ValueError(f"case {case_id!r} {field} contains duplicates")
        retrieved = set(case["retrieved_chunk_ids"])
        citations = set(case["expected_citation_chunk_ids"])
        if retrieved - chunks.keys() or citations - retrieved:
            raise ValueError(
                f"case {case_id!r} has unknown or unretrieved citation chunk IDs"
            )
        fact_support = case.get("fact_support")
        if not isinstance(fact_support, dict) or set(fact_support) != set(
            case["expected_facts"]
        ):
            raise ValueError(
                f"case {case_id!r} fact_support must annotate every expected fact"
            )
        for fact, annotation in fact_support.items():
            if not isinstance(annotation, dict) or set(annotation) != {
                "status",
                "chunk_ids",
            }:
                raise ValueError(
                    f"case {case_id!r} has invalid fact annotation for {fact!r}"
                )
            status = annotation["status"]
            chunk_ids = annotation["chunk_ids"]
            if (
                status not in {"supported", "withheld"}
                or not isinstance(chunk_ids, list)
                or any(not isinstance(chunk_id, str) for chunk_id in chunk_ids)
            ):
                raise ValueError(
                    f"case {case_id!r} has invalid evidence annotation for {fact!r}"
                )
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ValueError(f"case {case_id!r} fact {fact!r} repeats evidence IDs")
            if status == "supported" and not chunk_ids:
                raise ValueError(
                    f"case {case_id!r} supported fact {fact!r} needs evidence"
                )
            if status == "withheld" and chunk_ids:
                raise ValueError(
                    f"case {case_id!r} withheld fact {fact!r} cannot cite evidence"
                )
            if any(chunk_id not in chunks for chunk_id in chunk_ids):
                raise ValueError(
                    f"case {case_id!r} fact {fact!r} references an unknown chunk"
                )
            if any(chunk_id not in retrieved for chunk_id in chunk_ids):
                raise ValueError(
                    f"case {case_id!r} fact {fact!r} references evidence not supplied"
                )
        unsupported_claims = case.get("unsupported_claims", [])
        if not isinstance(unsupported_claims, list) or any(
            not isinstance(claim, str) or not claim.strip()
            for claim in unsupported_claims
        ):
            raise ValueError(
                f"case {case_id!r} unsupported_claims must be non-empty text"
            )
        if (
            not isinstance(case.get("scripted_answer"), str)
            or not case["scripted_answer"].strip()
        ):
            raise ValueError(f"case {case_id!r} needs scripted_answer")
        judgments = case.get("judgments")
        if not isinstance(judgments, dict) or set(judgments) != {
            "supported",
            "contradicted",
            "unsupported",
        }:
            raise ValueError(
                f"case {case_id!r} needs supported/contradicted/unsupported judgments"
            )
        if any(
            not isinstance(items, list)
            or any(not isinstance(item, str) for item in items)
            for items in judgments.values()
        ):
            raise ValueError(f"case {case_id!r} judgments must be lists of text")


def evaluate_answer_case(
    case: Mapping[str, Any],
    chunks: Mapping[str, Any],
    abstention_phrases: Sequence[str],
) -> AnswerCaseResult:
    """Run one fixture case through RAGService and compute transparent checks."""
    results = [
        _to_retrieval_result(chunks[chunk_id])
        for chunk_id in case["retrieved_chunk_ids"]
    ]
    response = RAGService(
        _ControlledRetriever(results), _ScriptedProvider(case["scripted_answer"])
    ).answer(RAGRequest(case["question"]))
    normalized = " ".join(response.answer.casefold().split())
    matched = sum(
        " ".join(fact.casefold().split()) in normalized
        for fact in case["expected_facts"]
    )
    supported_facts = [
        fact
        for fact, annotation in case["fact_support"].items()
        if annotation["status"] == "supported"
    ]
    matched_supported = sum(
        " ".join(fact.casefold().split()) in normalized for fact in supported_facts
    )
    unsupported_matches = tuple(
        claim
        for claim in case.get("unsupported_claims", [])
        if " ".join(claim.casefold().split()) in normalized
    )
    expected = [
        _citation(chunks[chunk_id]) for chunk_id in case["expected_citation_chunk_ids"]
    ]
    actual = [_response_citation(citation) for citation in response.citations]
    if len(actual) != len(set(actual)):
        raise ValueError(f"case {case['case_id']!r} produced duplicate citation tuples")
    precision, recall = retrieved_source_attribution_metrics(expected, actual)
    phrase_match = any(phrase.casefold() in normalized for phrase in abstention_phrases)
    return AnswerCaseResult(
        case["case_id"],
        len(case["expected_facts"]),
        matched,
        len(supported_facts),
        matched_supported,
        precision,
        recall,
        unsupported_matches,
        phrase_match,
    )


def _to_retrieval_result(chunk: Mapping[str, Any]) -> Any:
    text = chunk["text"]
    metadata = DocumentMetadata(
        id=chunk["document_id"],
        filename=chunk["filename"],
        content_type="text/plain",
        storage_path=f"{chunk['document_id']}.txt",
        size_bytes=len(text.encode("utf-8")),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        extracted_text_length=len(text),
    )
    return RetrievalResult(
        chunk_id=chunk["chunk_id"],
        document_id=chunk["document_id"],
        chunk_index=chunk["chunk_index"],
        text=text,
        document=metadata,
        similarity=1.0,
    )


def _citation(chunk: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        chunk["document_id"],
        chunk["chunk_id"],
        chunk["filename"],
        chunk["chunk_index"],
    )


def _response_citation(citation: Citation) -> tuple[str, str, str, int]:
    return (
        citation.document_id,
        citation.chunk_id,
        citation.filename,
        citation.chunk_index,
    )
