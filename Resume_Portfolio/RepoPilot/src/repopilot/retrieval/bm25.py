"""Small dependency-free BM25 index for retrieval experiments."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]")
_INDEXABLE_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json"}


def tokenize(text: str) -> list[str]:
    """Tokenize source while preserving identifiers and splitting snake_case."""
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if "_" in match:
            tokens.extend(part for part in match.split("_") if part)
    return tokens


@dataclass(frozen=True, slots=True)
class CodeDocument:
    """One indexed repository file."""

    path: str
    text: str
    terms: Counter[str]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A ranked lexical retrieval result."""

    path: str
    score: float
    excerpt: str


class BM25CodeIndex:
    """An inspectable BM25 baseline used before embeddings or rerankers."""

    def __init__(self, documents: list[CodeDocument], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.average_length = (
            sum(sum(document.terms.values()) for document in documents) / len(documents)
            if documents
            else 0.0
        )
        postings: dict[str, int] = defaultdict(int)
        for document in documents:
            for term in document.terms:
                postings[term] += 1
        self.document_frequency = dict(postings)

    @classmethod
    def build(
        cls,
        root: Path,
        *,
        max_file_bytes: int = 500_000,
        path_filter: Callable[[Path], bool] | None = None,
    ) -> BM25CodeIndex:
        documents: list[CodeDocument] = []
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in _INDEXABLE_SUFFIXES
                or ".git" in path.parts
                or path.stat().st_size > max_file_bytes
                or (path_filter is not None and not path_filter(path))
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            terms = Counter(tokenize(f"{path.relative_to(root).as_posix()}\n{text}"))
            documents.append(CodeDocument(path.relative_to(root).as_posix(), text, terms))
        return cls(documents)

    def search(self, query: str, *, limit: int = 5, excerpt_chars: int = 800) -> list[RetrievalHit]:
        if limit <= 0 or excerpt_chars <= 0:
            raise ValueError("limit and excerpt_chars must be positive")
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.documents:
            return []
        total_documents = len(self.documents)
        hits: list[RetrievalHit] = []
        for document in self.documents:
            length = sum(document.terms.values())
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = document.terms.get(term, 0)
                if term_frequency == 0:
                    continue
                frequency = self.document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1.0 + (total_documents - frequency + 0.5) / (frequency + 0.5)
                )
                normalization = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.average_length, 1.0)
                )
                score += (
                    query_frequency
                    * inverse_frequency
                    * (term_frequency * (self.k1 + 1.0) / normalization)
                )
            if score > 0:
                hits.append(
                    RetrievalHit(
                        document.path,
                        score,
                        self._excerpt(document.text, query_terms, max_chars=excerpt_chars),
                    )
                )
        return sorted(hits, key=lambda item: (-item.score, item.path))[:limit]

    @staticmethod
    def _excerpt(text: str, query_terms: Counter[str], *, max_chars: int = 800) -> str:
        lines = text.splitlines()
        best_index = 0
        best_score = -1
        for index, line in enumerate(lines):
            line_terms = set(tokenize(line))
            score = sum(query_terms[term] for term in line_terms if term in query_terms)
            if score > best_score:
                best_index = index
                best_score = score
        start = max(0, best_index - 2)
        excerpt = "\n".join(
            f"{index + 1}: {lines[index]}"
            for index in range(start, min(len(lines), best_index + 4))
        )
        return excerpt[:max_chars]
