"""Frozen, offline CodeRAG-Bench RepoEval retrieval runs for R1.

This adapter deliberately evaluates retrieval only.  It uses CodeRAG-Bench's
materialised ``queries.jsonl``, ``corpus.jsonl``, and qrels files and keeps
the query split separate from the shared public corpus.  No model is fitted
on development or validation queries; the final-holdout assignments are
loaded but rejected by the command-line runner until the pre-registered
candidate is selected.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from repopilot.evaluation.retrieval import QrelRetrievalSummary, qrel_retrieval_metrics
from repopilot.evidence.protocol import (
    DatasetInstance,
    ExperimentReceipt,
    FrozenManifest,
    build_grouped_split_manifest,
    canonical_sha256,
    sha256_file,
)
from repopilot.retrieval.adaptive import AdaptiveCodeHybrid, CodeChunk, python_symbol_chunks
from repopilot.retrieval.bm25 import BM25CodeIndex, CodeDocument, tokenize
from repopilot.retrieval.dense import DenseRetriever

RunVariant = Literal[
    "bm25_top_k",
    "hybrid_rrf",
    "routed_hybrid_rrf",
    "routed_hybrid_with_symbol_test_expansion",
    "adaptive_code_hybrid",
]
_VARIANTS: tuple[RunVariant, ...] = (
    "bm25_top_k",
    "hybrid_rrf",
    "routed_hybrid_rrf",
    "routed_hybrid_with_symbol_test_expansion",
    "adaptive_code_hybrid",
)
_SPLITS = ("development", "validation", "final_holdout")


@dataclass(frozen=True, slots=True)
class RepoEvalQuery:
    """One CodeRAG RepoEval query keyed by the official task ID."""

    query_id: str
    repository: str
    text: str


@dataclass(frozen=True, slots=True)
class RepoEvalDataset:
    """Validated materialised CodeRAG RepoEval data."""

    root: Path
    queries: dict[str, RepoEvalQuery]
    documents_by_repository: dict[str, tuple[CodeDocument, ...]]
    chunks_by_repository: dict[str, tuple[CodeChunk, ...]]
    qrels: dict[str, dict[str, int]]
    sha256: str


def load_repoeval_dataset(root: Path) -> RepoEvalDataset:
    """Load the upstream materialisation and preserve its opaque document IDs."""

    repositories = sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "queries.jsonl").is_file()
        and (directory / "corpus.jsonl").is_file()
        and (directory / "qrels" / "test.tsv").is_file()
    )
    if not repositories:
        raise ValueError(f"no complete RepoEval repositories found under {root}")
    queries: dict[str, RepoEvalQuery] = {}
    documents_by_repository: dict[str, tuple[CodeDocument, ...]] = {}
    chunks_by_repository: dict[str, tuple[CodeChunk, ...]] = {}
    qrels: dict[str, dict[str, int]] = {}
    digested_files: list[dict[str, str]] = []
    for directory in repositories:
        repository = directory.name
        query_path = directory / "queries.jsonl"
        corpus_path = directory / "corpus.jsonl"
        qrels_path = directory / "qrels" / "test.tsv"
        for path in (query_path, corpus_path, qrels_path):
            digested_files.append(
                {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            )
        for raw in _jsonl(query_path):
            query_id = _required_string(raw, "_id", query_path)
            if query_id in queries:
                raise ValueError(f"duplicate query ID in RepoEval data: {query_id}")
            queries[query_id] = RepoEvalQuery(
                query_id=query_id,
                repository=repository,
                text=_required_string(raw, "text", query_path),
            )
        documents: list[CodeDocument] = []
        chunks: list[CodeChunk] = []
        for raw in _jsonl(corpus_path):
            document_id = _required_string(raw, "_id", corpus_path)
            text = _required_string(raw, "text", corpus_path)
            display_path = _document_path(raw, document_id)
            documents.append(
                CodeDocument(
                    path=document_id,
                    text=text,
                    terms=Counter(tokenize(f"{display_path}\n{text}")),
                )
            )
            chunks.append(_code_chunk(document_id, display_path, text))
        if not documents or len({item.path for item in documents}) != len(documents):
            raise ValueError(f"invalid or duplicate corpus IDs in {corpus_path}")
        documents_by_repository[repository] = tuple(documents)
        chunks_by_repository[repository] = tuple(chunks)
        qrel_lines = qrels_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(qrel_lines, start=1):
            if not line.strip() or line.startswith("query-id\t"):
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"invalid qrel at {qrels_path}:{line_number}")
            query_id, document_id, raw_grade = fields
            if query_id not in queries or queries[query_id].repository != repository:
                raise ValueError(
                    f"qrel references unknown or wrong-repo query at {qrels_path}:{line_number}"
                )
            if document_id not in {document.path for document in documents}:
                raise ValueError(f"qrel references unknown corpus ID at {qrels_path}:{line_number}")
            try:
                grade = int(raw_grade)
            except ValueError as error:
                raise ValueError(f"non-integer qrel at {qrels_path}:{line_number}") from error
            qrels.setdefault(query_id, {})[document_id] = grade
    missing_qrels = sorted(set(queries) - set(qrels))
    if missing_qrels:
        raise ValueError(f"RepoEval queries without qrels, first IDs: {missing_qrels[:3]}")
    return RepoEvalDataset(
        root=root,
        queries=queries,
        documents_by_repository=documents_by_repository,
        chunks_by_repository=chunks_by_repository,
        qrels=qrels,
        sha256=canonical_sha256(sorted(digested_files, key=lambda item: item["path"])),
    )


def build_repoeval_manifest(dataset: RepoEvalDataset) -> FrozenManifest:
    """Freeze a query-level 20/40/40 split, stratified by source repository."""

    return build_grouped_split_manifest(
        dataset_name="CodeRAG-Bench/repoeval_repo",
        dataset_version="CodeRAG-Bench@f9e100ca9ed94b8f1983b356ae81966e30210cf4",
        dataset_sha256=dataset.sha256,
        instances=[
            DatasetInstance(
                instance_id=query.query_id,
                group_ids=(f"query:{query.query_id}",),
                stratum=query.repository,
            )
            for query in dataset.queries.values()
        ],
        notes=(
            "Query-level split; each public repository corpus remains available to every split, "
            "as required for retrieval evaluation.",
            "No retriever parameters are trained from development or validation query outcomes.",
            "Do not execute final_holdout before the R1 candidate is selected on validation.",
        ),
    )


def load_manifest(path: Path) -> FrozenManifest:
    """Parse a manifest produced by the evidence protocol without accepting extras."""

    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return FrozenManifest(
        schema_version=int(raw["schema_version"]),
        dataset_name=str(raw["dataset_name"]),
        dataset_version=str(raw["dataset_version"]),
        dataset_sha256=str(raw["dataset_sha256"]),
        split_algorithm=str(raw["split_algorithm"]),
        assignments={name: tuple(raw["assignments"][name]) for name in _SPLITS},
        strata_counts={name: dict(raw["strata_counts"][name]) for name in _SPLITS},
        notes=tuple(raw.get("notes", [])),
    )


def run_repoeval_variant(
    dataset: RepoEvalDataset,
    manifest: FrozenManifest,
    *,
    split: Literal["development", "validation", "final_holdout"],
    variant: RunVariant,
    embedding_cache: Path | None = None,
) -> dict[str, object]:
    """Evaluate one frozen split and return rankings plus qrel metrics.

    The function is deliberately in-memory; the CLI below is responsible for
    immutable raw-output and receipt persistence.
    """

    if manifest.dataset_sha256 != dataset.sha256:
        raise ValueError("manifest dataset hash does not match the materialised RepoEval dataset")
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported R1 variant: {variant}")
    selected_ids = manifest.assignments[split]
    selected_queries = [dataset.queries[query_id] for query_id in selected_ids]
    dense_enabled = variant != "bm25_top_k"
    embedding_by_id: dict[str, list[float]] = {}
    if dense_enabled:
        if embedding_cache is None:
            raise ValueError("embedding_cache is required for hybrid R1 variants")
        embedding_by_id = load_or_build_embedding_cache(dataset, embedding_cache)
    indexes: dict[str, AdaptiveCodeHybrid] = {}
    for repository, documents in dataset.documents_by_repository.items():
        lexical = BM25CodeIndex(list(documents))
        dense = (
            DenseRetriever(
                list(documents),
                document_embeddings=[embedding_by_id[document.path] for document in documents],
            )
            if dense_enabled
            else None
        )
        indexes[repository] = AdaptiveCodeHybrid(
            dataset.chunks_by_repository[repository], lexical=lexical, dense=dense
        )
    use_routing = variant in {
        "routed_hybrid_rrf",
        "routed_hybrid_with_symbol_test_expansion",
        "adaptive_code_hybrid",
    }
    use_expansion = variant in {
        "routed_hybrid_with_symbol_test_expansion",
        "adaptive_code_hybrid",
    }
    use_budget = variant == "adaptive_code_hybrid"
    rankings: dict[str, list[str]] = {}
    routes: dict[str, str] = {}
    latency_ms: list[float] = []
    for query in selected_queries:
        start = time.perf_counter()
        route, hits = indexes[query.repository].retrieve(
            query.text,
            limit=10,
            use_hybrid=dense_enabled,
            use_routing=use_routing,
            expand=use_expansion,
            expansion_seed_limit=5 if use_expansion else None,
        )
        if use_budget:
            hits = indexes[query.repository].select_budget(hits, context_token_budget=4096)
        latency_ms.append((time.perf_counter() - start) * 1000.0)
        rankings[query.query_id] = [item.chunk.chunk_id for item in hits]
        routes[query.query_id] = route.kind
    summary = qrel_retrieval_metrics(
        rankings,
        {query_id: dataset.qrels[query_id] for query_id in selected_ids},
        cutoffs=(5, 10),
    )
    return {
        "schema_version": 1,
        "dataset": "CodeRAG-Bench/repoeval_repo",
        "dataset_sha256": dataset.sha256,
        "manifest_sha256": manifest.sha256,
        "split": split,
        "variant": variant,
        "queries": len(selected_queries),
        "metrics": _summary_dict(summary),
        "latency_ms": _latency_summary(latency_ms),
        "route_counts": dict(sorted(Counter(routes.values()).items())),
        "rankings": {query_id: rankings[query_id] for query_id in sorted(rankings)},
    }


def load_or_build_embedding_cache(
    dataset: RepoEvalDataset, cache_path: Path
) -> dict[str, list[float]]:
    """Return content-validated Ollama embeddings, creating a cache exactly once."""

    documents = [
        document
        for repository in sorted(dataset.documents_by_repository)
        for document in dataset.documents_by_repository[repository]
    ]
    expected = {document.path: sha256_file_content(document.text) for document in documents}
    if cache_path.exists():
        raw = cast(dict[str, Any], json.loads(cache_path.read_text(encoding="utf-8")))
        cached_vectors = raw.get("vectors")
        if (
            raw.get("schema_version") != 1
            or raw.get("model") != "nomic-embed-text"
            or raw.get("dataset_sha256") != dataset.sha256
            or not isinstance(cached_vectors, dict)
            or set(cached_vectors) != set(expected)
        ):
            raise ValueError(
                f"embedding cache is incompatible with the frozen RepoEval corpus: {cache_path}"
            )
        loaded: dict[str, list[float]] = {}
        for document_id, content_hash in expected.items():
            entry = cached_vectors[document_id]
            if not isinstance(entry, dict) or entry.get("text_sha256") != content_hash:
                raise ValueError(f"embedding cache text hash mismatch for {document_id}")
            vector = entry.get("embedding")
            if (
                not isinstance(vector, list)
                or not vector
                or not all(isinstance(value, (int, float)) for value in vector)
            ):
                raise ValueError(f"embedding cache vector invalid for {document_id}")
            loaded[document_id] = [float(value) for value in vector]
        return loaded
    built_vectors: dict[str, dict[str, object]] = {}
    for batch in _batches(documents, size=16):
        embeddings = _ollama_embed_batch([document.text[:8000] for document in batch])
        if len(embeddings) != len(batch):
            raise RuntimeError("Ollama returned a different number of embeddings than requested")
        for document, embedding in zip(batch, embeddings, strict=True):
            built_vectors[document.path] = {
                "text_sha256": expected[document.path],
                "embedding": embedding,
            }
    cache_payload = {
        "schema_version": 1,
        "model": "nomic-embed-text",
        "endpoint": "http://127.0.0.1:11434/api/embed",
        "dataset_sha256": dataset.sha256,
        "vectors": built_vectors,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    return {
        document_id: [float(value) for value in cast(list[float], entry["embedding"])]
        for document_id, entry in built_vectors.items()
    }


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"non-object JSONL record at {path}:{line_number}")
        yield cast(dict[str, Any], raw)


def _required_string(raw: Mapping[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty {key!r} in {path}")
    return value


def _document_path(raw: Mapping[str, Any], fallback: str) -> str:
    metadata = raw.get("metadata")
    first = metadata[0] if isinstance(metadata, list) and metadata else metadata
    if isinstance(first, dict):
        fpath = first.get("fpath_tuple")
        if isinstance(fpath, list) and fpath and all(isinstance(item, str) for item in fpath):
            return "/".join(fpath)
    title = raw.get("title")
    return title if isinstance(title, str) and title.strip() else fallback


def _code_chunk(document_id: str, display_path: str, text: str) -> CodeChunk:
    symbols: set[str] = set()
    references: set[str] = set()
    if display_path.endswith(".py"):
        for parsed in python_symbol_chunks(display_path, text):
            symbols.update(parsed.symbols)
            references.update(parsed.references)
    return CodeChunk(
        chunk_id=document_id,
        path=display_path,
        start_line=1,
        end_line=max(1, len(text.splitlines())),
        text=text,
        symbols=tuple(sorted(symbols)),
        references=tuple(sorted(references)),
    )


def _ollama_embed_batch(inputs: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": "nomic-embed-text", "input": inputs}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = cast(dict[str, object], json.loads(response.read()))
    except urllib.error.URLError as error:
        raise RuntimeError("Ollama nomic-embed-text endpoint is unavailable") from error
    embeddings = raw.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError("Ollama returned no embeddings")
    result: list[list[float]] = []
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Ollama returned an invalid embedding")
        result.append([float(value) for value in embedding])
    return result


def _batches(items: list[CodeDocument], *, size: int) -> Iterable[list[CodeDocument]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _summary_dict(summary: QrelRetrievalSummary) -> dict[str, object]:
    return {
        "ndcg_at": {str(cutoff): value for cutoff, value in summary.ndcg_at.items()},
        "recall_at": {str(cutoff): value for cutoff, value in summary.recall_at.items()},
        "mrr_at": {str(cutoff): value for cutoff, value in summary.mrr_at.items()},
    }


def _latency_summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(samples)
    return {
        "mean": statistics.fmean(samples),
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256_file_content(text: str) -> str:
    return canonical_sha256({"text": text})


def _git_value(args: list[str], fallback: str) -> str:
    completed = subprocess.run(args, capture_output=True, check=False, text=True)
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return fallback


def _write_run(
    dataset: RepoEvalDataset,
    manifest: FrozenManifest,
    *,
    split: Literal["development", "validation", "final_holdout"],
    variant: RunVariant,
    output: Path,
    embedding_cache: Path | None,
) -> None:
    receipt_path = output.with_suffix(".receipt.json")
    if output.exists() or receipt_path.exists():
        raise ValueError("refusing to overwrite R1 raw output or receipt")
    result = run_repoeval_variant(
        dataset, manifest, split=split, variant=variant, embedding_cache=embedding_cache
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = ExperimentReceipt(
        schema_version=1,
        run_id=output.stem,
        variant=variant,
        manifest_sha256=manifest.sha256,
        source_commit=_git_value(["git", "rev-parse", "HEAD"], "unknown"),
        environment={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "working_tree_dirty": str(
                bool(_git_value(["git", "status", "--porcelain"], ""))
            ).lower(),
        },
        fixed_controls={
            "dense_model": "nomic-embed-text" if variant != "bm25_top_k" else "disabled",
            "expansion_seed_limit": (
                5 if "expansion" in variant or variant == "adaptive_code_hybrid" else None
            ),
            "rrf_k": 60 if variant != "bm25_top_k" else None,
            "top_k": 10,
            "context_token_budget": 4096 if variant == "adaptive_code_hybrid" else None,
        },
        raw_output_sha256=sha256_file(output),
        notes=(
            "Offline retrieval metric only; this result is not a SWE-bench resolved-rate claim.",
            "The final_holdout command remains manually gated by the pre-registered "
            "candidate rule.",
        ),
    )
    receipt.write_once(receipt_path)


def main(argv: list[str] | None = None) -> int:
    """Run the narrow R1 materialisation, manifest, and retrieval workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    create = subcommands.add_parser("build-manifest")
    create.add_argument("--dataset-root", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--split", choices=_SPLITS, required=True)
    run.add_argument(
        "--final-holdout-approved",
        action="store_true",
        help="required explicit guard after the candidate has been frozen on validation",
    )
    run.add_argument("--variant", choices=_VARIANTS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--embedding-cache", type=Path)
    args = parser.parse_args(argv)
    dataset = load_repoeval_dataset(args.dataset_root)
    if args.command == "build-manifest":
        manifest = build_repoeval_manifest(dataset)
        manifest.write_once(args.manifest)
        print(
            json.dumps({"manifest": str(args.manifest), "sha256": manifest.sha256}, sort_keys=True)
        )
        return 0
    manifest = load_manifest(args.manifest)
    if args.split == "final_holdout" and not args.final_holdout_approved:
        parser.error("final_holdout requires --final-holdout-approved after candidate freeze")
    _write_run(
        dataset,
        manifest,
        split=args.split,
        variant=args.variant,
        output=args.output,
        embedding_cache=args.embedding_cache,
    )
    print(
        json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
