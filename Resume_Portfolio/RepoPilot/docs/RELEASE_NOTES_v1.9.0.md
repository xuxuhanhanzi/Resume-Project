# RepoPilot v1.9.0

## P0 — installation and entrypoint diagnostics

- `repopilot doctor` now reports the running launcher, the venv launcher, imported package path and
  the `repopilot` command resolved from `PATH`. It diagnoses the common Anaconda/old-venv shadowing
  failure without exposing credentials, invoking pip, deleting metadata or contacting a provider.
- It also identifies stale `~epopilot-*.dist-info` metadata left by an interrupted editable install and
  gives only safe manual remediation guidance.

## P1 — evidence-led coding evaluation

- Added `repopilot eval diagnose-swe <predictions.jsonl> <resolved_results.json>`.
- The offline command classifies each completed SWE-style task as `resolved`, `empty_patch`,
  `patch_apply_failed`, `regression`, `fail_to_pass_failed`, or `unresolved_other`. It does not launch
  Docker, call a model, or read evaluator-only test patches.

## P2 — everyday CLI interaction

- Added static Tab completion for common slash-command arguments and local typo suggestions for unknown
  slash commands. A misspelled command is never sent to a model.
- Added repeatable `--add-dir DIRECTORY` support. Explicit external folders are exposed under aliases
  such as `@extra1` exclusively through bounded read/list/search tools. They cannot grant external
  writes, commands, Git, LSP, tests, symlink escapes or path traversal.

## P3 — background task receipts

- Background children now create redacted, durable task receipts. `/tasks` survives a CLI restart and
  truthfully marks previously running children as `unavailable`; the agent never claims it can reattach.
- `/task-log <id>` reads bounded output only while the current CLI owns a live child. `/retry <id>` is an
  explicit, separately confirmed new launch and is unavailable when command arguments were redacted.

## P4 — controlled MCP hardening

- Added `repopilot mcp doctor`: a zero-launch, zero-request check of stdio launchers and loopback HTTP
  configuration.
- The stdio transport now drains bounded stderr, includes redacted failure context, preserves minimal
  Windows process variables, and still does not inherit provider credentials. MCP tools remain
  high-risk and require `--mcp` plus workspace trust.

## Verification

- `226 passed, 1 skipped` (`pytest -q`)
- `ruff check .` passes
- `mypy src tests scripts` passes strictly (188 source files)
- `pip check` passes
