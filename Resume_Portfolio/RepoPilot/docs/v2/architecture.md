# RepoPilot v2 Architecture

## Purpose

RepoPilot v2 evolves the v1 task-oriented runtime into a session-oriented CLI
coding agent without weakening the deterministic evaluation harness.

## Runtime boundary

```text
                     AgentKernel
                    /           \
                   /             \
          TaskRuntime         SessionRuntime
          (benchmark)        (interactive CLI)
```

`AgentKernel` owns one bounded model/tool turn. It receives a model request,
executes approved tool calls, and emits ordered runtime events. It does not
create workspaces, decide task completion, persist sessions, or render a CLI.

`TaskRuntime` owns the v1-compatible `PublicTaskSpec` lifecycle: baseline
snapshots, deterministic verification, repair loops, benchmark contracts, and
run-artifact persistence.

`SessionRuntime` owns persistent user conversations, interactive permissions,
context projection/compaction, project instructions, session forks, and
user-triggered verification. It must never depend on hidden benchmark data.

`LocalOpenAICompatibleProvider` remains loopback-only. `DeepSeekProvider` is a
separate opt-in cloud boundary: its endpoint is fixed to DeepSeek's documented
chat-completions URL, model identifiers are allowlisted, and its credential is
read only from a named process environment variable. It does not accept an
arbitrary remote base URL.

## Event contract

The kernel exposes ordered events rather than making a UI wait for a complete
run. Initial event kinds are:

- `turn_started`
- `model_call_started`
- `model_call_completed`
- `tool_call_proposed`
- `permission_requested`
- `tool_call_started`
- `tool_call_completed`
- `verification_started`
- `verification_completed`
- `turn_completed`
- `turn_failed`
- `turn_cancelled`

Events are data, not terminal strings. The CLI, trace recorder, hooks, and
`--output-format jsonl` one-shot transport render or persist the same event
stream. JSONL output keeps stdout machine-readable by emitting `event` records
followed by one `result` record.

## Safety invariants

- Every tool call is validated and routed through a policy decision before it
  reaches a runner.
- General shell execution uses tokenized argv and `shell=False`; all interactive
  executions require an approval even in `accept_edits` mode.
- An incomplete side-effecting operation is never replayed automatically after
  a crash. It must be surfaced to the user for confirmation.
- The CLI persists an explicit workspace-trust decision before creating or
  resuming a project session. Project instructions remain untrusted data even
  in a trusted workspace and cannot override the policy layer.
- Session identifiers are canonical UUIDs, mutable operations take an
  exclusive per-session lease, and atomic state writes use unique temporary
  paths. Durable transcripts redact common credential forms before writing.
- A turn has one cooperative cancellation token. Cancelling an awaitable command
  stops the launched local process (and its Windows child tree where the host
  supports `taskkill`), clears unexecuted pending calls, and records a durable
  `turn_cancelled` event. The next turn starts from that safe checkpoint.
- Agent control-plane paths (`REPOPILOT.md`, `AGENTS.md`, `.repopilot/`, and
  deployment/build controls) require explicit approval even in `accept_edits`.

## Implemented interactive capabilities

- `ShellTool` validates bounded argv input, uses the existing `CommandRunner`,
  and cannot invoke a command shell.
- Git observations are limited to `status`, `log`, and safe-ref `show`.
- `ProcessManager` powers tokenized `start_background`, read-only
  `background_status`, and high-risk `terminate_background` tools. Process
  handles are scoped to the current `SessionRuntime` process and are never
  silently reattached after a restart; closing the runtime terminates every
  still-running process it owns.
- `ContextManager` stores an append-only transcript while sending the model a
  bounded projection plus a deterministic, persisted handoff summary. Its
  projection never leaves a native assistant tool-call or tool result orphaned:
  a clipped partial exchange is omitted as a unit.
- The local OpenAI-compatible provider serializes retained tool exchanges using
  standard assistant `tool_calls` followed by `tool` messages. A legacy or
  orphaned observation is explicitly labelled untrusted user data instead of
  being sent as an invalid native tool message.
- `VerificationEngine` is available through `/verify`; it uses the same runner
  and interactive approval policy, and persists structured verification events.

## Advanced extension boundary

- Project skills have metadata-first discovery and persistent explicit
  activation. Their bodies are bounded and treated as untrusted instructions.
- Hooks are JSON-declared audit notices only. They receive events but cannot
  execute a process, import code, access a network, or bypass a policy.
- `--mcp` is explicit opt-in. Configured stdio servers run without a shell;
  every remote tool is namespaced and classified `HIGH_RISK` before reaching
  `AgentKernel`, and every started transport is closed through the interactive
  command's single lifecycle boundary.
- Explore, test-analysis, and review subagents use selected evidence only,
  receive no tools, and fail if they return a tool call.
- Worktree creation is high-risk, workspace-contained, creation-only, and
  never removes a path. Git commits are high-risk and Git staging remains
  subject to protected-path policy.

## Compatibility invariants

- `repopilot run --task ...` remains supported throughout the migration.
- `PublicTaskSpec`, `ToolCall`, `ToolResult`, `ModelRequest`, and
  `ModelResponse` remain provider-neutral contracts.
- Benchmark hidden tests remain outside model context and the project
  workspace.
