# RepoPilot v1.11.0

## P11 — Windows first-run and recovery acceptance

- `scripts/setup_windows.ps1` now rejects the ambiguous `-WithDev -WithService` combination.
- `scripts/acceptance_windows.ps1` parses the project-local path, launcher, offline `doctor`,
  shell-init and MCP configuration, and runs focused Windows/recovery tests by default. It never
  edits `$PROFILE`, credentials or project configuration, and never constructs a provider.

## P12 — independent synthetic coding-development fixtures

- Added the versioned public `repopilot-synthetic-dev-v1` manifest and `repopilot eval coding-dev plan`.
- `coding-dev run` materializes fresh artifact-only workspaces and emits redacted funnel JSONL.
  It requires explicit cloud `--provider`, `--trust` and `--allow-fixture-edits`; it is a development
  diagnostic, not a benchmark or capability claim. No model request was made for this release.

## P13 — repair evidence and cloud-context boundary

- Added `/evidence`, an offline session view of policy facts, the latest verification outcome and
  the latest recorded workspace change inventory.
- Complete verification logs remain in redacted local session receipts. New verification tool
  messages forwarded to future model turns contain only a structural outcome receipt, so logs are
  not automatically sent to a cloud provider. `/repair` requires a persisted failing verification.

## P14 — durable supervisor diagnosis

- Supervisor submissions support a no-secret label and a 0.1–3600 second wall-time limit.
- The per-project queue is capped at 20 queued jobs. `supervisor show <id>` reads status, timeout
  and failure information locally. Launch failures and timeouts become durable failed receipts;
  interrupted jobs still require a separate, explicit retry.

## P15 — explicit MCP health/audit protocol

- `mcp probe <name> --timeout-seconds 1..120` records a redacted local receipt only after the user
  explicitly trusts and probes the named configured server.
- `mcp history [name]` is offline and displays those receipts. Consecutive successful probes compare
  the exposed, post-policy tool names and report capability additions/removals. MCP child processes
  retain their minimal environment and do not inherit cloud API credentials.

## Verification

The release is verified with the project venv using `ruff check .`, strict `mypy src tests scripts`,
`pytest -q`, `pip check`, `git diff --check`, the PowerShell parser check for both Windows scripts,
and offline CLI smoke commands. The precise aggregate test count is recorded in `docs/STATUS.md`.
