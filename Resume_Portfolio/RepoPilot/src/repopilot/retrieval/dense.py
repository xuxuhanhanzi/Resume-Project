"""Dense retrieval via Ollama's batch-compatible embedding endpoint."""

from __future__ import annotations

import json
import math
import urllib.request
from typing import cast

from repopilot.retrieval.bm25 import CodeDocument, RetrievalHit


class DenseRetriever:
    """Cosine-similarity retriever backed by Ollama /api/embeddings.

    Embeddings are computed once per chunk at construction time and cached
    in-memory.  This is intentionally dependency-free (no numpy) so it runs
    inside the project's minimal venv.
    """

    def __init__(
        self,
        documents: list[CodeDocument],
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        document_embeddings: list[list[float]] | None = None,
    ) -> None:
        self.documents = documents
        self.model = model
        self.base_url = base_url.rstrip("/")
        if document_embeddings is not None and len(document_embeddings) != len(documents):
            raise ValueError("document_embeddings must have one vector per document")
        self._doc_embeddings = (
            [list(vector) for vector in document_embeddings]
            if document_embeddings is not None
            else [self._embed(doc.text) for doc in documents]
        )
        self._doc_norms = [self._norm(emb) for emb in self._doc_embeddings]

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": text[:8000]}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        raw_embeddings = cast(list[list[float]], data.get("embeddings", []))
        emb = raw_embeddings[0] if len(raw_embeddings) == 1 else []
        if not emb:
            raise RuntimeError(f"empty embedding from {self.model} for text len={len(text)}")
        return emb

    @staticmethod
    def _norm(vec: list[float]) -> float:
        return math.sqrt(sum(v * v for v in vec))

    def search(
        self, query: str, *, limit: int = 5, excerpt_chars: int = 1_600
    ) -> list[RetrievalHit]:
        if limit <= 0 or not self.documents:
            return []
        query_emb = self._embed(query)
        query_norm = self._norm(query_emb)
        if query_norm == 0:
            return []
        scored: list[tuple[float, int]] = []
        for idx, doc_emb in enumerate(self._doc_embeddings):
            doc_norm = self._doc_norms[idx]
            if doc_norm == 0:
                continue
            dot = sum(a * b for a, b in zip(query_emb, doc_emb, strict=True))
            score = dot / (query_norm * doc_norm)
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[RetrievalHit] = []
        for score, idx in scored[:limit]:
            doc = self.documents[idx]
            hits.append(RetrievalHit(doc.path, score, doc.text[:excerpt_chars]))
        return hits
