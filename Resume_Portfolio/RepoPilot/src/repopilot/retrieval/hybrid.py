"""Hybrid retrieval combining BM25 and dense via Reciprocal Rank Fusion."""

from __future__ import annotations

from repopilot.retrieval.bm25 import BM25CodeIndex, CodeDocument, RetrievalHit
from repopilot.retrieval.dense import DenseRetriever


class HybridRetriever:
    """Fuse BM25 and dense rankings using Reciprocal Rank Fusion (RRF).

    RRF(d) = sum_r 1 / (k + rank_r(d))

    where *k* is a smoothing constant (default 60) and *rank_r(d)* is the
    1-based rank of document *d* in retriever *r*'s result list.  Documents
    not returned by a retriever contribute zero for that retriever.
    """

    def __init__(
        self,
        documents: list[CodeDocument],
        dense_retriever_cls: type[DenseRetriever],
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        dense_model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.documents = documents
        self.rrf_k = rrf_k
        self.bm25 = BM25CodeIndex(documents, k1=bm25_k1, b=bm25_b)
        self.dense = dense_retriever_cls(documents, model=dense_model, base_url=base_url)

    def search(
        self, query: str, *, limit: int = 5, excerpt_chars: int = 1_600
    ) -> list[RetrievalHit]:
        if limit <= 0 or not self.documents:
            return []
        over_fetch = max(limit * 3, 15)
        bm25_hits = self.bm25.search(query, limit=over_fetch, excerpt_chars=excerpt_chars)
        dense_hits = self.dense.search(query, limit=over_fetch, excerpt_chars=excerpt_chars)

        rrf_scores: dict[str, float] = {}
        excerpts: dict[str, str] = {}

        for rank, hit in enumerate(bm25_hits, start=1):
            rrf_scores[hit.path] = rrf_scores.get(hit.path, 0.0) + 1.0 / (self.rrf_k + rank)
            excerpts[hit.path] = hit.excerpt

        for rank, hit in enumerate(dense_hits, start=1):
            rrf_scores[hit.path] = rrf_scores.get(hit.path, 0.0) + 1.0 / (self.rrf_k + rank)
            if hit.path not in excerpts:
                excerpts[hit.path] = hit.excerpt

        ranked = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [RetrievalHit(path, score, excerpts.get(path, "")) for path, score in ranked]
