"""Dependency-free qrel metrics for the frozen R1 retrieval comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QrelRetrievalSummary:
    """Macro-averaged metrics for one ranking over one fixed qrel set."""

    queries: int
    ndcg_at: dict[int, float]
    recall_at: dict[int, float]
    mrr_at: dict[int, float]


def qrel_retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    *,
    cutoffs: tuple[int, ...] = (5, 10),
) -> QrelRetrievalSummary:
    """Compute NDCG, binary Recall, and MRR at K from positive graded qrels."""

    if not qrels or set(rankings) != set(qrels):
        raise ValueError("rankings and qrels must share the same non-empty query IDs")
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive")
    ndcgs = {cutoff: [] for cutoff in cutoffs}
    recalls = {cutoff: [] for cutoff in cutoffs}
    mrrs = {cutoff: [] for cutoff in cutoffs}
    for query_id in sorted(qrels):
        relevant = {
            document_id: float(grade)
            for document_id, grade in qrels[query_id].items()
            if float(grade) > 0.0
        }
        if not relevant:
            raise ValueError(f"query {query_id} has no positive qrels")
        ranking = list(rankings[query_id])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"query {query_id} ranking contains duplicate document IDs")
        for cutoff in cutoffs:
            top = ranking[:cutoff]
            dcg = sum(
                (2.0 ** relevant.get(document_id, 0.0) - 1.0) / math.log2(index + 1)
                for index, document_id in enumerate(top, start=1)
            )
            ideal = sorted(relevant.values(), reverse=True)[:cutoff]
            ideal_dcg = sum(
                (2.0**grade - 1.0) / math.log2(index + 1)
                for index, grade in enumerate(ideal, start=1)
            )
            ndcgs[cutoff].append(dcg / ideal_dcg if ideal_dcg else 0.0)
            recalls[cutoff].append(len(set(top) & set(relevant)) / len(relevant))
            rank = next(
                (
                    index
                    for index, document_id in enumerate(top, start=1)
                    if document_id in relevant
                ),
                None,
            )
            mrrs[cutoff].append(0.0 if rank is None else 1.0 / rank)
    return QrelRetrievalSummary(
        queries=len(qrels),
        ndcg_at={cutoff: sum(values) / len(values) for cutoff, values in ndcgs.items()},
        recall_at={cutoff: sum(values) / len(values) for cutoff, values in recalls.items()},
        mrr_at={cutoff: sum(values) / len(values) for cutoff, values in mrrs.items()},
    )
