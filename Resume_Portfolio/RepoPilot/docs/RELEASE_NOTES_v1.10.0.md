# RepoPilot v1.10.0

## P5 — safe Windows bootstrap

- Added `scripts/setup_windows.ps1`: it creates or reuses only the project-local `.venv`, installs
  RepoPilot, verifies the launcher and prints a session-local `shell-init` command. It never deletes
  an environment, edits PowerShell `$PROFILE`, stores credentials or changes global Python.

## P6 — coding development-funnel evidence

- Added `repopilot eval coding-funnel <records.jsonl>` and a strict local protocol for the observable
  stages `localized → nonempty patch → applied → verified`.
- This is development-only, runs no model/tests/patches, redacts patch content from output, and is
  explicitly separated from the frozen five-task SWE-bench-Live holdout. No claim of a new coding
  score is made.

## P7–P8 — repair discipline and session visibility

- A recoverable `apply_patch` conflict now requires a standalone fresh `read_file` before a retry;
  one ignored correction safely stops the turn before another patch is executed.
- `/status` now shows local session progress, plan state, latest persisted verification outcome,
  durable task-receipt counts, MCP state and input mode without starting tests or contacting a provider.

## P9 — explicit local supervisor

- Added `repopilot supervisor {serve,submit,list,log,stop,retry}` for user-authorized, shell-free jobs
  scoped to one trusted project. The supervisor is a separate foreground process, so closing the
  interactive CLI does not cancel its child.
- Commands and logs are redacted before durable persistence. A supervisor restart marks old running
  jobs `interrupted`; it never replays them. Retrying is a separate explicit action.

## P10 — controlled MCP capability inspection

- Per-server `allow_tools` / `deny_tools` policies can reduce exposed remote tools while preserving
  high-risk governance for every remaining MCP tool.
- `repopilot --trust mcp probe <name>` starts or contacts only the named configured server, prints
  discovered capability names, then closes it. `mcp doctor` remains zero-launch and zero-request.

## Verification

Verified on 2026-08-23 with the project venv: `ruff check .` passed; strict
`mypy src tests scripts` passed on 192 sources; `pytest -q` passed 238 tests
with one intentional live-Docker skip; `pip check` reported no broken
requirements; and `git diff --check` was clean. The venv still emits its
previously diagnosed stale `~epopilot-*.dist-info` warning, but `pip check` is
healthy; this release does not delete user environment metadata.
