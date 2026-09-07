"""Adaptive code-evidence retrieval for the R1 ablation matrix.

The implementation keeps every component independently switchable: hybrid
fusion, deterministic query routing, symbol/test expansion, and fixed-budget
selection.  It therefore supports the R1-A through R1-D comparisons without
silently changing more than one primary variable.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

from repopilot.retrieval.protocols import Retriever

_PATH = re.compile(r"(?:^|[\s'\"(])((?:[\w.-]+/)*[\w.-]+\.(?:py|js|ts|java|go|rs))(?:$|[\s'\",):])")
_STACK = re.compile(r"(?:traceback|file \".+\", line \d+|at .+\(.+:\d+:\d+\))", re.IGNORECASE)
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """A stable code evidence unit; chunk_id is also the retriever document path."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    symbols: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.chunk_id
            or not self.path
            or self.start_line < 1
            or self.end_line < self.start_line
            or not self.text.strip()
        ):
            raise ValueError("code chunk has invalid identity or line range")

    @property
    def is_test(self) -> bool:
        parts = PurePosixPath(self.path).parts
        return (
            self.path.startswith("test_")
            or "tests" in parts
            or any(part.startswith("test_") for part in parts)
        )


@dataclass(frozen=True, slots=True)
class QueryRoute:
    """Recorded deterministic query classification and RRF weighting."""

    kind: str
    lexical_weight: int
    dense_weight: int


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """Ranked evidence with an auditable origin and fixed-token estimate."""

    chunk: CodeChunk
    score: float
    origin: str
    estimated_tokens: int


def route_code_query(query: str) -> QueryRoute:
    """Prefer lexical evidence for paths, symbols, and stack traces."""

    if _STACK.search(query) or _PATH.search(query):
        return QueryRoute("trace_or_path", lexical_weight=2, dense_weight=1)
    identifiers = _IDENTIFIER.findall(query)
    if any("_" in item or any(char.isupper() for char in item[1:]) for item in identifiers):
        return QueryRoute("identifier", lexical_weight=2, dense_weight=1)
    return QueryRoute("natural_language", lexical_weight=1, dense_weight=1)


class AdaptiveCodeHybrid:
    """RRF retrieval plus optional graph expansion and budget selection."""

    def __init__(
        self,
        chunks: Iterable[CodeChunk],
        *,
        lexical: Retriever,
        dense: Retriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        chunk_list = tuple(chunks)
        if not chunk_list or len({chunk.chunk_id for chunk in chunk_list}) != len(chunk_list):
            raise ValueError("chunks must be non-empty and have unique IDs")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self._symbol_to_chunks, self._test_neighbours = _build_adjacency(chunk_list)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        use_hybrid: bool = True,
        use_routing: bool = True,
        expand: bool = False,
        expansion_seed_limit: int | None = None,
    ) -> tuple[QueryRoute, list[RetrievalEvidence]]:
        """Return only ranked chunk IDs known to this index."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        if expansion_seed_limit is not None and not 0 < expansion_seed_limit <= limit:
            raise ValueError("expansion_seed_limit must be in the range [1, limit]")
        route = route_code_query(query) if use_routing else QueryRoute("fixed", 1, 1)
        overfetch = max(limit * 3, 15)
        lexical_hits = self.lexical.search(query, limit=overfetch)
        dense_hits = self.dense.search(query, limit=overfetch) if use_hybrid and self.dense else []
        scores: dict[str, float] = defaultdict(float)
        origins: dict[str, str] = {}
        for rank, hit in enumerate(lexical_hits, start=1):
            if hit.path in self.chunks:
                scores[hit.path] += route.lexical_weight / (self.rrf_k + rank)
                origins[hit.path] = "lexical"
        for rank, hit in enumerate(dense_hits, start=1):
            if hit.path in self.chunks:
                scores[hit.path] += route.dense_weight / (self.rrf_k + rank)
                origins[hit.path] = "hybrid" if hit.path in origins else "dense"
        seed_limit = expansion_seed_limit if expand and expansion_seed_limit is not None else limit
        ranked = [
            RetrievalEvidence(
                self.chunks[chunk_id],
                score,
                origins[chunk_id],
                _estimate_tokens(self.chunks[chunk_id].text),
            )
            for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
                :seed_limit
            ]
        ]
        return route, self._expand(ranked, limit=limit) if expand else ranked

    def select_budget(
        self,
        candidates: Iterable[RetrievalEvidence],
        *,
        context_token_budget: int,
    ) -> list[RetrievalEvidence]:
        """Select non-duplicate evidence under a fixed, recorded approximation."""

        if context_token_budget <= 0:
            raise ValueError("context_token_budget must be positive")
        selected: list[RetrievalEvidence] = []
        remaining = context_token_budget
        seen_content: set[str] = set()
        path_count: dict[str, int] = defaultdict(int)
        remaining_candidates = list(candidates)
        while remaining_candidates:
            evidence = min(
                remaining_candidates,
                key=lambda item: (
                    -(item.score / (1.0 + path_count[item.chunk.path])),
                    item.chunk.path,
                    item.chunk.start_line,
                ),
            )
            remaining_candidates.remove(evidence)
            content_hash = hashlib.sha256(evidence.chunk.text.encode("utf-8")).hexdigest()
            if content_hash in seen_content or evidence.estimated_tokens > remaining:
                continue
            selected.append(evidence)
            remaining -= evidence.estimated_tokens
            seen_content.add(content_hash)
            path_count[evidence.chunk.path] += 1
        return selected

    def _expand(self, ranked: list[RetrievalEvidence], *, limit: int) -> list[RetrievalEvidence]:
        """Add one-hop symbol and test neighbours without reshuffling base hits."""

        expanded = list(ranked)
        included = {item.chunk.chunk_id for item in expanded}
        for evidence in ranked:
            neighbours: list[tuple[str, str]] = []
            for symbol in evidence.chunk.symbols + evidence.chunk.references:
                neighbours.extend(
                    (chunk_id, "symbol_neighbour")
                    for chunk_id in self._symbol_to_chunks.get(symbol, ())
                )
            neighbours.extend(
                (chunk_id, "test_neighbour")
                for chunk_id in self._test_neighbours.get(evidence.chunk.chunk_id, ())
            )
            for chunk_id, origin in sorted(set(neighbours)):
                if chunk_id in included:
                    continue
                chunk = self.chunks[chunk_id]
                expanded.append(
                    RetrievalEvidence(
                        chunk,
                        evidence.score * 0.5,
                        origin,
                        _estimate_tokens(chunk.text),
                    )
                )
                included.add(chunk_id)
                if len(expanded) >= limit:
                    return expanded
        return expanded


def python_symbol_chunks(path: str, text: str) -> list[CodeChunk]:
    """Produce AST-bounded Python chunks; unparsable files have a lexical fallback."""

    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [
            CodeChunk(
                chunk_id=f"{path}#L1-L{max(1, len(lines))}",
                path=path,
                start_line=1,
                end_line=max(1, len(lines)),
                text=text,
            )
        ]
    chunks: list[CodeChunk] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno)
        source = "\n".join(lines[start - 1 : end])
        references = tuple(sorted(set(_IDENTIFIER.findall(source)) - {node.name}))
        chunks.append(
            CodeChunk(
                chunk_id=f"{path}#L{start}-L{end}",
                path=path,
                start_line=start,
                end_line=end,
                text=source,
                symbols=(node.name,),
                references=references,
            )
        )
    return chunks or [
        CodeChunk(
            chunk_id=f"{path}#L1-L{max(1, len(lines))}",
            path=path,
            start_line=1,
            end_line=max(1, len(lines)),
            text=text,
        )
    ]


def _build_adjacency(
    chunks: tuple[CodeChunk, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    symbols: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for symbol in chunk.symbols:
            symbols[symbol].add(chunk.chunk_id)
    symbol_to_chunks = {key: tuple(sorted(value)) for key, value in symbols.items()}
    test_neighbours: dict[str, tuple[str, ...]] = {}
    tests = tuple(chunk for chunk in chunks if chunk.is_test)
    for chunk in chunks:
        if chunk.is_test:
            continue
        related = [
            test.chunk_id
            for test in tests
            if set(chunk.symbols) & (set(test.symbols) | set(test.references))
            or PurePosixPath(chunk.path).stem in test.path
        ]
        if related:
            test_neighbours[chunk.chunk_id] = tuple(sorted(related))
    return symbol_to_chunks, test_neighbours


def _estimate_tokens(text: str) -> int:
    """Fixed public approximation used only for a cross-variant budget gate."""

    return max(1, math.ceil(len(text) / 4))
