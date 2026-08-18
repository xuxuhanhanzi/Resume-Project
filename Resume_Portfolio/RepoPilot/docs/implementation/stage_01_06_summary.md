# RepoPilot Stage 1–6 Implementation Summary

This is an implementation/evidence record. Teaching notes will be written only after the
runtime, local-model experiment, and quality evidence are frozen.

## Stage 1 — Minimal Agent Runtime

Implemented provider-neutral messages, model requests/responses, structured tool calls/results,
explicit state, hard budgets, a bounded ReAct loop, ScriptedProvider, ToolRegistry, JSONL Trace,
and atomic Checkpoint. Completion is not a model action: a deterministic verifier owns it.

Evidence: contract/budget tests and the offline integrated Agent test.

## Stage 2 — Local model and coding tools

Implemented a loopback-only OpenAI-compatible provider for locally served Qwen-class models.
It normalizes native tool calls and JSON AgentAction fallback. Coding tools cover bounded file
listing/reading/search, Python AST symbols, exact replacement Patch, immutable tests, Diff, and
failure inspection. Arbitrary shell is absent from the default surface.

Evidence: provider parsing, tool closed-loop, and scripted calculator repair tests. The formal
experiment model is frozen: qwen2.5:7b (Q4_K_M), digest 845dbda0ea48..., served via Ollama.

## Stage 3 — Context, retrieval, memory, and skills

Implemented history trimming, tool-output compression, trust-boundary instructions, a BM25
code index, Agentic retrieval tool, session facts, keyword episodic memory, and SKILL.md
progressive disclosure. Embeddings and a vector database are deferred until lexical retrieval
has a frozen baseline.

Evidence: BM25 ranking, memory bounds/search, skill discovery/activation, and context tests.

## Stage 4 — MCP and orchestration patterns

Implemented an educational JSON-RPC MCP subset with initialize/tools-list/tools-call, an
in-process transport, discovery client, and remote Tool Adapter. Added a transparent Planner,
parallel read-only tool execution, and an isolated read-only Reviewer Agent-as-Tool.

Evidence: MCP discovery/call/adapter test, side-effect rejection in parallel execution, and
reviewer isolation test. This subset is not advertised as complete MCP wire compatibility.

## Stage 5 — Recovery and security

Implemented pending-action checkpoints, an idempotency journal, recoverable model/tool retry,
human-approval state, structural policy decisions, path canonicalization, output limits, a
trusted-fixture runner, and a fail-closed Docker sandbox runner. Docker policy includes no
network, read-only root, non-root user, dropped capabilities, no-new-privileges, PID/memory/CPU
limits, and a bounded writable workspace mount.

Evidence: recovery replay, traversal rejection, untrusted-local refusal, Docker-command policy,
approval, budget, and redaction tests. Docker daemon 29.6.1 is verified available (network none),
three SWE official images are materialized, and SWE Agent smoke experiments have been officially
evaluated (resolved=0/3).

## Stage 6 — Evaluation and integrated demos

Implemented outcome/cost/latency/safety aggregation, File Recall@k, MRR, fixed-task ablation
checks, Public/Evaluator Task separation, HiddenTestGrader, ten micro Bug Fix cases, and CLI
demos for scripted end-to-end execution, MCP, permission, and recovery.

The ten tasks are development fixtures, not a claim of broad coding-agent ability. Formal three-
repeat experiments have been completed for FRAMES (B1=16.7%) and DABench (B1=90.0% — the
10-task smoke result later superseded by 71.4% / 25-of-35 on the frozen val35 set) with frozen
model hash, quantization, budget, and complete run artifacts. SWE-bench-Live smoke=0/3.

## Artifact contract

Each run writes a human-inspectable directory containing `baseline.json`, `checkpoint.json`,
`tool_journal.json`, and `events.jsonl`. Episodic memory is stored separately so task recovery
does not depend on an ever-growing model context.
