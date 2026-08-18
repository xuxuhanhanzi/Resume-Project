"""Retriever, Reranker, and ContextCompressor protocols for P8A ablation."""

from __future__ import annotations

from typing import Protocol

from repopilot.retrieval.bm25 import CodeDocument, RetrievalHit


class Retriever(Protocol):
    """Abstract retriever interface for single-document chunk search."""

    def search(
        self, query: str, *, limit: int = 5, excerpt_chars: int = 1_600
    ) -> list[RetrievalHit]:
        """Return ranked hits for *query* from this retriever's chunk set."""
        ...


class Reranker(Protocol):
    """Re-orders a candidate hit list using a stronger model."""

    def rerank(
        self, query: str, hits: list[RetrievalHit], *, limit: int = 5
    ) -> list[RetrievalHit]: ...


class ContextCompressor(Protocol):
    """Compresses retrieved context while preserving factual content."""

    def compress(self, query: str, context: str) -> str: ...


def chunk_document(
    text: str, *, chunk_chars: int = 2_000, overlap: int = 250, uri: str = ""
) -> list[CodeDocument]:
    """Split *text* into overlapping CodeDocument chunks with deterministic IDs."""
    chunks: list[CodeDocument] = []
    from collections import Counter

    from repopilot.retrieval.bm25 import tokenize

    step = chunk_chars - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars]
        if not chunk.strip():
            continue
        chunk_id = f"{uri}#chars={start}-{start + len(chunk)}"
        chunks.append(CodeDocument(chunk_id, chunk, Counter(tokenize(chunk))))
    return chunks


class FirstChunkRetriever:
    """A2 ablation: return the first chunk(s) without any search.

    This simulates a "no retrieval" baseline where the agent always sees
    the first fixed-length segment of each oracle document, regardless of
    the query.  Comparing this to BM25/Dense/Hybrid isolates the value of
    active retrieval versus passive context.
    """

    def __init__(self, documents: list[CodeDocument]) -> None:
        self.documents = documents

    def search(
        self, query: str, *, limit: int = 5, excerpt_chars: int = 1_600
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(doc.path, 0.0, doc.text[:excerpt_chars]) for doc in self.documents[:limit]
        ]
