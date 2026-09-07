from __future__ import annotations

from dataclasses import dataclass

from repopilot.retrieval.adaptive import AdaptiveCodeHybrid, CodeChunk, python_symbol_chunks
from repopilot.retrieval.bm25 import RetrievalHit


@dataclass
class FixtureRetriever:
    ranking: list[str]

    def search(
        self, query: str, *, limit: int = 5, excerpt_chars: int = 1_600
    ) -> list[RetrievalHit]:
        del query, excerpt_chars
        return [
            RetrievalHit(path, float(len(self.ranking) - index), "")
            for index, path in enumerate(self.ranking[:limit])
        ]


def test_adaptive_hybrid_routes_trace_and_expands_to_test_neighbour() -> None:
    implementation = CodeChunk(
        "src/math.py#L1-L2",
        "src/math.py",
        1,
        2,
        "def subtract(left, right):\n    return left + right",
        symbols=("subtract",),
    )
    test = CodeChunk(
        "tests/test_math.py#L1-L2",
        "tests/test_math.py",
        1,
        2,
        "from src.math import subtract\nassert subtract(2, 1) == 1",
        references=("subtract",),
    )
    other = CodeChunk("src/html.py#L1-L1", "src/html.py", 1, 1, "def render(): pass")
    retriever = AdaptiveCodeHybrid(
        [implementation, test, other],
        lexical=FixtureRetriever([implementation.chunk_id, other.chunk_id]),
        dense=FixtureRetriever([other.chunk_id, implementation.chunk_id]),
    )

    route, hits = retriever.retrieve('File "src/math.py", line 2', limit=3, expand=True)

    assert route.kind == "trace_or_path"
    assert hits[0].chunk.chunk_id == implementation.chunk_id
    assert {item.chunk.chunk_id for item in hits} >= {implementation.chunk_id, test.chunk_id}


def test_budget_selector_deduplicates_content_and_python_chunker_has_symbols() -> None:
    chunks = python_symbol_chunks("src/example.py", "def alpha():\n    return 1\n")
    duplicate = CodeChunk(
        "src/duplicate.py#L1-L2",
        "src/duplicate.py",
        1,
        2,
        chunks[0].text,
        symbols=("beta",),
    )
    retriever = AdaptiveCodeHybrid(
        [chunks[0], duplicate],
        lexical=FixtureRetriever([chunks[0].chunk_id, duplicate.chunk_id]),
    )
    _, evidence = retriever.retrieve("alpha", limit=2, use_hybrid=False)
    selected = retriever.select_budget(evidence, context_token_budget=100)

    assert chunks[0].symbols == ("alpha",)
    assert len(selected) == 1


def test_expansion_ignores_unresolved_identifier_references() -> None:
    chunk = CodeChunk(
        "src/example.py#L1-L2",
        "src/example.py",
        1,
        2,
        "def alpha():\n    return missing_name",
        symbols=("alpha",),
        references=("missing_name",),
    )
    retriever = AdaptiveCodeHybrid([chunk], lexical=FixtureRetriever([chunk.chunk_id]))

    _, hits = retriever.retrieve("alpha", limit=1, use_hybrid=False, expand=True)

    assert [item.chunk.chunk_id for item in hits] == [chunk.chunk_id]
