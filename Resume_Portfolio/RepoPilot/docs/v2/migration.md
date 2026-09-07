# RepoPilot v2 Migration Record

## Stage 0 baseline

- v1 baseline tag: `v1.0.4`
- repository entry point: `main`
- historical source quality gate: 90 passing tests and one opt-in Docker test
  skipped by default
- current developer environments must install the package from this repository
  root; a stale editable installation from an older directory is not a valid
  baseline

Before treating a result as reproducible, verify that `pip show repopilot`
reports this repository as the editable project location and run the regression
gates in `compatibility.md`. Recreate a Python environment only after any old
environment has been deliberately retired by its owner.

## Stage 1 migration

1. Add provider-neutral event and turn contracts.
2. Move model/tool execution from `AgentRuntime` into `AgentKernel`.
3. Introduce `TaskRuntime` for the task lifecycle and verification loop.
4. Make `AgentRuntime` delegate to `TaskRuntime` until downstream callers have
   migrated.
5. Add characterization tests for event order, tool replay, interruption, and
   v1 command compatibility.

No existing benchmark, evaluation, orchestration, or API module is moved in
Stage 1.

## Implemented Alpha slice

The first implementation slice adds an event-emitting `AgentKernel`, keeps the
v1 task path behind its compatible `AgentRuntime` / `TaskRuntime` seam, and
introduces `ProjectWorkspace`, `SessionStore`, `SessionRuntime`, and the default
interactive CLI entry point. Session turns use the existing structured file and
search tools with a manual approval default; generic Shell and process handling
remain intentionally out of scope until their command-policy contract lands.

## Implemented Beta core slice

- Session state now has bounded model-context projection, deterministic
  compaction, `/context`, `/compact`, resume, and independent fork support.
- The CLI persists explicit workspace trust outside the repository. It loads
  `REPOPILOT.md`, `AGENTS.md`, and `.repopilot/rules/*.md` as bounded untrusted
  context only.
- `run_shell` accepts only argv arrays, never shell strings, and follows the
  existing `PermissionEngine -> CommandRunner` execution path.
- Git status/log/show are read-only tools. A bounded `ProcessManager` supports
  approved background start/status/terminate actions in the active session.
- `/verify` runs explicit or conservative project-default verification through
  the same approval boundary and writes structured session events.

Remaining Beta work includes interactive repair orchestration and richer
language-specific verification.

## Implemented advanced extension slice

- Project-local skills are deferred, bounded, selected by explicit slash
  command or prompt relevance, and persisted per session.
- Project hooks are intentionally declarative notice-only extensions with
  transcript audit entries; arbitrary project hook execution remains excluded.
- The MCP client now supports validated stdio configuration and line-framed
  JSON-RPC. Server startup requires `--mcp` after workspace trust, and remote
  tools are always namespaced plus high-risk.
- Read-only Explore/Test subagents complement the existing reviewer with
  isolated evidence-only contexts.
- Approved Git stage/commit and workspace-contained creation-only worktrees
  support a conservative collaborative workflow.

Remaining product hardening focuses on richer interactive repair policy,
language-specific verification strategies, full interoperability testing with
external MCP implementations, and longer session/safety evaluation suites.

## Product hardening slice

- Default verification is now marker-based for Python, package.json scripts,
  Cargo, Go, and Maven. Script bodies are never parsed into shell commands.
- Verification reports are appended to the durable conversation as structured
  tool observations. `/repair` is explicitly opt-in and only activates after
  the latest verification fails.
- The stdio MCP transport validates JSON-RPC version and response ID, and its
  registry validates remote names before namespacing them.
- A 50-turn scripted session integration test verifies compaction, restart,
  resume, and lossless durable transcript behavior. Existing integration and
  safety suites remain part of the mandatory quality gate.

## Follow-up security and reliability hardening

- Session loading rejects non-UUID identities, session mutations are serialized
  by an exclusive lease, and stale locks are recovered only when their recorded
  process is no longer present.
- Project instruction and extension control-plane files cannot be silently
  changed in `accept_edits`; case variants are covered on Windows too.
- Verification events now travel through the same hook/audit dispatch path as
  tool events. MCP child servers close from one `finally` boundary on normal
  exit, errors, and interruption.
- Transcript event data is redacted before persistence, and `list_files` walks
  only its requested depth and maximum result count instead of scanning the
  complete repository first.

## Runtime-control and automation boundary

- A cooperative `CancellationToken` now flows from a session turn through the
  kernel, tool context, and command runner. Cancelled turns are checkpointed as
  `cancelled`, emit a durable audit event, and clear unexecuted pending calls so
  resume never silently replays a side effect.
- The local runner uses awaitable subprocesses rather than a thread-wrapped
  synchronous process. Cancelling the awaitable or timing out stops the child;
  on Windows it also requests termination of the owned child tree. Docker CLI
  invocations use this same boundary.
- `SessionRuntime.aclose()` now closes every session-scoped `ProcessManager`.
  The interactive CLI invokes that lifecycle hook from its `finally` path, so
  normal exit, errors, and interrupts do not leave RepoPilot-managed background
  processes behind.
- One-shot CLI calls support `--output-format jsonl`. It emits structured
  runtime lifecycle records and a final result on stdout for IDE and script
  integration, without claiming model-token streaming support.
- Conversation messages now retain the structured tool-call identity that
  produced each tool result. Context clipping preserves a complete native tool
  exchange or drops it as a unit; the local provider emits the standard
  assistant `tool_calls` / `tool` pair whenever that relationship is present.
- DeepSeek is available as an explicit cloud Provider (`--provider deepseek`),
  without weakening the loopback-only local Provider. It fixes the documented
  endpoint, accepts the current `deepseek-v4-flash` / `deepseek-v4-pro` model
  IDs, and requires an environment-variable Key before any network request.
  Offline tests cover its authorization header, endpoint, missing-Key failure,
  and CLI boundary; live calls remain a deliberate user action.
