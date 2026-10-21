"""Small offline metrics for evaluating semantic retrieval results."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from secondbrain.application.retrieval import RetrievalResult


class RetrievalEvaluator(Protocol):
    """The part of semantic retrieval needed by the metric runner."""

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievalResult]: ...


@dataclass(frozen=True)
class EvaluationQuery:
    """One question and its expected supporting chunk identities."""

    query_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval metrics over answerable evaluation questions."""

    query_count: int
    answerable_query_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    no_relevant_chunk_queries: int
    k: int


def evaluate_retrieval(
    retriever: RetrievalEvaluator,
    queries: Sequence[EvaluationQuery],
    *,
    k: int,
) -> RetrievalEvaluationReport:
    """Calculate Recall@k and MRR, excluding unanswerable cases.

    A query with no expected chunk is unanswerable and is excluded from both
    ranking metrics. The missing-relevant count includes answerable queries for
    which none of the expected chunks appears in the top ``k`` results.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    answerable = [query for query in queries if query.relevant_chunk_ids]
    if not answerable:
        raise ValueError("evaluation requires at least one answerable query")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    no_relevant = 0
    for query in answerable:
        if len(query.relevant_chunk_ids) != len(set(query.relevant_chunk_ids)):
            raise ValueError(
                f"query {query.query_id!r} contains duplicate relevant chunk IDs"
            )
        results = retriever.retrieve(query.question, top_k=k)
        relevant_ids = set(query.relevant_chunk_ids)
        ranked_ids = [result.chunk_id for result in results]
        hits = len(relevant_ids.intersection(ranked_ids))
        recalls.append(hits / len(relevant_ids))
        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )
        reciprocal_ranks.append(
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        )
        if hits == 0:
            no_relevant += 1

    return RetrievalEvaluationReport(
        query_count=len(queries),
        answerable_query_count=len(answerable),
        recall_at_k=sum(recalls) / len(recalls),
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
        no_relevant_chunk_queries=no_relevant,
        k=k,
    )
