"""Embedding-based reranker for P8A R3 experiment."""

from __future__ import annotations

import json
import math
import urllib.request
from typing import cast

from repopilot.retrieval.bm25 import RetrievalHit


class EmbeddingReranker:
    """Rerank retrieval hits using query-chunk cosine similarity.

    Unlike RRF fusion (which only uses ranks), this reranker uses the
    continuous cosine similarity between the query embedding and each
    chunk embedding, producing a finer-grained relevance ordering.
    """

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text[:8000]}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        emb = cast(list[float], data.get("embedding", []))
        if not emb:
            raise RuntimeError(f"empty embedding from {self.model}")
        return emb

    @staticmethod
    def _norm(vec: list[float]) -> float:
        return math.sqrt(sum(v * v for v in vec))

    def rerank(self, query: str, hits: list[RetrievalHit], *, limit: int = 5) -> list[RetrievalHit]:
        if not hits or limit <= 0:
            return []
        query_emb = self._embed(query)
        query_norm = self._norm(query_emb)
        if query_norm == 0:
            return hits[:limit]
        scored: list[tuple[float, RetrievalHit]] = []
        for hit in hits:
            chunk_emb = self._embed(hit.excerpt[:8000])
            chunk_norm = self._norm(chunk_emb)
            if chunk_norm == 0:
                scored.append((0.0, hit))
                continue
            dot = sum(a * b for a, b in zip(query_emb, chunk_emb, strict=True))
            score = dot / (query_norm * chunk_norm)
            scored.append((score, hit))
        scored.sort(key=lambda item: (-item[0], item[1].path))
        return [hit for _, hit in scored[:limit]]
