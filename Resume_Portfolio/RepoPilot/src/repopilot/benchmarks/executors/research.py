"""Offline document-grounded executor for research benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from repopilot.benchmarks.contracts import (
    ArtifactRef,
    BenchmarkTask,
    EnvironmentHandle,
    EvidenceRef,
    PreparedTask,
    SharedRunMetrics,
    TaskOutput,
)
from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProvider
from repopilot.retrieval.bm25 import BM25CodeIndex, CodeDocument, tokenize
from repopilot.retrieval.protocols import Retriever


@dataclass(frozen=True, slots=True)
class OracleDocumentExecutorConfig:
    """Frozen settings for the FRAMES oracle-document smoke baseline."""

    chunks_per_document: int = 2
    chunk_chars: int = 2_000
    chunk_overlap_chars: int = 250
    temperature: float = 0.0
    max_output_tokens: int = 128
    reasoning_output_tokens: int = 512
    plan_queries: bool = False
    reasoned_answer: bool = False
    review_answer: bool = False
    retriever_type: str = "bm25"  # "bm25" | "dense" | "hybrid"
    dense_model: str = "nomic-embed-text"
    base_url: str = "http://127.0.0.1:11434"
    use_reranker: bool = False
    query_rewrite: bool = False
    chunks_per_document_override: int = 0  # 0 = use chunks_per_document

    def __post_init__(self) -> None:
        if self.chunks_per_document <= 0 or self.chunk_chars <= 0:
            raise ValueError("retrieval limits must be positive")
        if not 0 <= self.chunk_overlap_chars < self.chunk_chars:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        if self.max_output_tokens <= 0 or self.reasoning_output_tokens <= 0:
            raise ValueError("output budget must be positive")


@dataclass(frozen=True, slots=True)
class _CorpusDocument:
    url: str
    path: Path
    sha256: str


class OracleDocumentModelExecutor:
    """Retrieve excerpts from frozen oracle pages, then call one fixed model."""

    def __init__(
        self,
        provider: ModelProvider,
        corpus_manifest: Path,
        config: OracleDocumentExecutorConfig | None = None,
        planner_provider: ModelProvider | None = None,
    ) -> None:
        self.provider = provider
        self.planner_provider = planner_provider or provider
        self.manifest_path = corpus_manifest.resolve(strict=True)
        self.config = config or OracleDocumentExecutorConfig()
        self.documents = self._load_manifest()

    def _load_manifest(self) -> dict[str, _CorpusDocument]:
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
            raise ValueError("corpus manifest must contain a document list")
        documents: dict[str, _CorpusDocument] = {}
        for item in raw["documents"]:
            if not isinstance(item, dict):
                raise ValueError("corpus document entry must be an object")
            url = str(item.get("url", ""))
            relative_path = str(item.get("path", ""))
            expected_hash = str(item.get("text_sha256", ""))
            path = (self.manifest_path.parent / relative_path).resolve(strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not url or digest != expected_hash:
                raise ValueError(f"corpus document hash mismatch: {url or relative_path}")
            documents[url] = _CorpusDocument(url, path, digest)
        return documents

    def _make_retriever(self, chunks: list[CodeDocument]) -> Retriever:
        """Build a retriever according to config.retriever_type."""
        rtype = self.config.retriever_type
        if rtype == "no_retrieval":
            from repopilot.retrieval.protocols import FirstChunkRetriever

            return FirstChunkRetriever(chunks)
        if rtype == "dense":
            from repopilot.retrieval.dense import DenseRetriever

            return DenseRetriever(
                chunks, model=self.config.dense_model, base_url=self.config.base_url
            )
        if rtype == "hybrid":
            from repopilot.retrieval.dense import DenseRetriever
            from repopilot.retrieval.hybrid import HybridRetriever

            return HybridRetriever(
                chunks,
                DenseRetriever,
                dense_model=self.config.dense_model,
                base_url=self.config.base_url,
            )
        return BM25CodeIndex(chunks)

    def _retrieve(
        self, task: BenchmarkTask, queries: dict[str, str] | None = None
    ) -> tuple[str, tuple[EvidenceRef, ...]]:
        context_parts: list[str] = []
        evidence: list[EvidenceRef] = []
        for source_number, asset in enumerate(task.assets, start=1):
            try:
                document = self.documents[asset.uri]
            except KeyError as error:
                raise ValueError(
                    f"oracle document is missing from frozen corpus: {asset.uri}"
                ) from error
            text = document.path.read_text(encoding="utf-8")
            chunks: list[CodeDocument] = []
            step = self.config.chunk_chars - self.config.chunk_overlap_chars
            for start in range(0, len(text), step):
                chunk = text[start : start + self.config.chunk_chars]
                if not chunk.strip():
                    continue
                chunk_id = f"{asset.uri}#chars={start}-{start + len(chunk)}"
                chunks.append(CodeDocument(chunk_id, chunk, Counter(tokenize(chunk))))
            retriever = self._make_retriever(chunks)
            query = (queries or {}).get(asset.uri, task.instruction)
            if self.config.use_reranker:
                from repopilot.retrieval.reranker import EmbeddingReranker

                reranker = EmbeddingReranker(
                    model=self.config.dense_model, base_url=self.config.base_url
                )
                raw_hits = retriever.search(
                    query,
                    limit=self.config.chunks_per_document * 4,
                    excerpt_chars=1_600,
                )
                hits = reranker.rerank(query, raw_hits, limit=self.config.chunks_per_document)
            else:
                hits = retriever.search(
                    query,
                    limit=self.config.chunks_per_document,
                    excerpt_chars=1_600,
                )
            for hit_number, hit in enumerate(hits, start=1):
                context_parts.append(
                    f"[SOURCE {source_number}.{hit_number}] {asset.uri}\n{hit.excerpt}"
                )
                evidence.append(EvidenceRef(asset.uri, hit.path.rsplit("#", 1)[-1]))
        return "\n\n".join(context_parts), tuple(evidence)

    async def execute(
        self, task: BenchmarkTask, prepared: PreparedTask, handle: EnvironmentHandle
    ) -> TaskOutput:
        if prepared.task != task or handle.work_dir != prepared.work_dir:
            raise ValueError("oracle-document executor received mismatched task environment")
        started_at = monotonic()
        input_tokens = 0
        output_tokens = 0
        recovery_count = 0
        planner_path: Path | None = None
        queries: dict[str, str] | None = None
        if self.config.plan_queries:
            queries = {}
            raw_responses: dict[str, str] = {}
            for asset in task.assets:
                title = asset.uri.rsplit("/", 1)[-1].replace("_", " ")
                planner_response = await self.planner_provider.complete(
                    ModelRequest(
                        messages=(
                            Message(
                                "system",
                                "Plan one focused lexical search for one offline "
                                "encyclopedia page.",
                            ),
                            Message(
                                "user",
                                f"Question:\n{task.instruction}\n\nCurrent page: {title}\n\n"
                                "Infer this page's role in the multi-hop chain. Return "
                                "only a short "
                                "search query for the exact fact needed from this page. Preserve "
                                "relations such as mother versus mother's maiden name.",
                            ),
                        ),
                        tools=(),
                        temperature=self.config.temperature,
                        max_output_tokens=96,
                    )
                )
                input_tokens += planner_response.usage.input_tokens
                output_tokens += planner_response.usage.output_tokens
                planned_query = planner_response.content.strip().strip('`"')
                raw_responses[asset.uri] = planner_response.content
                if planned_query:
                    queries[asset.uri] = planned_query
            recovery_count = len(task.assets) - len(queries)
            planner_path = prepared.work_dir / "query_plan.json"
            planner_path.write_text(
                json.dumps(
                    {
                        "raw_responses": raw_responses,
                        "parsed_queries": queries or {},
                        "fallback_sources": recovery_count,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if self.config.query_rewrite and queries:
            for uri in list(queries):
                rewrite_resp = await self.planner_provider.complete(
                    ModelRequest(
                        messages=(
                            Message(
                                "system",
                                "Rewrite the search query to find different "
                                "relevant facts from the same page.",
                            ),
                            Message(
                                "user",
                                f"Original query: {queries[uri]}\n\nProvide one rewritten query:",
                            ),
                        ),
                        tools=(),
                        temperature=self.config.temperature,
                        max_output_tokens=96,
                    )
                )
                input_tokens += rewrite_resp.usage.input_tokens
                output_tokens += rewrite_resp.usage.output_tokens
                rewritten = rewrite_resp.content.strip().strip('`"')
                if rewritten:
                    queries[uri] = rewritten
        context, evidence = self._retrieve(task, queries)
        answer_instruction = (
            "Reason through the multi-hop chain step by step. End with one line in the exact "
            "form FINAL: <short answer>. Do not infer relatives from a person's own given or "
            "middle names; use explicit source statements."
            if self.config.reasoned_answer
            else "Use the excerpts to solve the question. Return only the shortest final answer."
        )
        prompt = (
            f"Question:\n{task.instruction}\n\nRetrieved source excerpts:\n{context}\n\n"
            f"{answer_instruction}"
        )
        response = await self.provider.complete(
            ModelRequest(
                messages=(
                    Message(
                        "system",
                        "You answer multi-hop questions only from the supplied offline sources.",
                    ),
                    Message("user", prompt),
                ),
                tools=(),
                temperature=self.config.temperature,
                max_output_tokens=(
                    self.config.reasoning_output_tokens
                    if self.config.reasoned_answer
                    else self.config.max_output_tokens
                ),
            )
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        final_answer = response.content.strip()
        if self.config.reasoned_answer:
            final_matches = re.findall(
                r"(?im)^\s*(?:FINAL|\*{0,2}Answer\*{0,2})\s*:\s*(.+?)\s*$",
                response.content,
            )
            if final_matches:
                final_answer = final_matches[-1].strip()
            else:
                recovery_count += 1
        reviewer_path: Path | None = None
        if self.config.review_answer:
            reviewer_response = await self.provider.complete(
                ModelRequest(
                    messages=(
                        Message(
                            "system",
                            "You are a critical verifier for multi-hop answers. Correct "
                            "factual, "
                            "ordering, relationship, and arithmetic mistakes using only the "
                            "sources.",
                        ),
                        Message(
                            "user",
                            f"Question:\n{task.instruction}\n\nSources:\n{context}\n\n"
                            f"Draft solution:\n{response.content}\n\n"
                            "Audit every step, correct the draft if needed, and end with exactly "
                            "one line: FINAL: <short answer>.",
                        ),
                    ),
                    tools=(),
                    temperature=self.config.temperature,
                    max_output_tokens=512,
                )
            )
            input_tokens += reviewer_response.usage.input_tokens
            output_tokens += reviewer_response.usage.output_tokens
            reviewed_matches = re.findall(
                r"(?im)^\s*FINAL\s*:\s*(.+?)\s*$", reviewer_response.content
            )
            if reviewed_matches:
                final_answer = reviewed_matches[-1].strip()
            else:
                recovery_count += 1
            reviewer_path = prepared.work_dir / "reviewer_response.json"
            reviewer_path.write_text(
                json.dumps(
                    {"model": reviewer_response.model, "content": reviewer_response.content},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        context_path = prepared.work_dir / "retrieval_context.txt"
        context_path.write_text(context + "\n", encoding="utf-8")
        response_path = prepared.work_dir / "model_response.json"
        response_path.write_text(
            json.dumps(
                {
                    "model": response.model,
                    "content": response.content,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_inputs = [
            (context_path, "text/plain"),
            (response_path, "application/json"),
        ]
        if planner_path is not None:
            artifact_inputs.append((planner_path, "application/json"))
        if reviewer_path is not None:
            artifact_inputs.append((reviewer_path, "application/json"))
        artifacts = tuple(
            ArtifactRef(
                str(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                media_type,
            )
            for path, media_type in artifact_inputs
        )
        return TaskOutput(
            final_answer=final_answer,
            structured_payload={"model": response.model, "retrieved_chunks": len(evidence)},
            artifacts=artifacts,
            evidence=evidence,
            run_metrics=SharedRunMetrics(
                iterations=(len(task.assets) + 1 if self.config.plan_queries else 1)
                + int(self.config.review_answer),
                tool_calls=len(task.assets),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                wall_seconds=monotonic() - started_at,
                recovery_count=recovery_count,
            ),
        )
