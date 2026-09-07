# RepoPilot v1.12.0

## P16–P17 — comparable synthetic development diagnostics

- A paid `coding-dev run` now writes `outcomes.redacted.jsonl` plus a distinct
  `run.receipt.json`. The receipt retains only suite digest, selected case IDs,
  provider/model names, bounded counters, statuses and redacted failure reasons.
- `repopilot eval coding-dev report <run-directory>` is fully offline. It validates the
  receipt against the funnel and proposes a small next-action backlog from observed failure
  categories. It never constructs a provider, runs a command, or reads raw traces.
- No cloud baseline was automatically run for this release. Consequently no agent prompt or
  runtime behavior is claimed to have improved; P17 changes remain evidence-gated.

## P18 — Windows launcher collision hardening

- `repopilot shell-doctor powershell` explains the active launcher/PATH relationship and prints
  a one-window `shell-init` command. It does not edit PATH, `$PROFILE`, Credential Manager or
  pip metadata.
- `scripts/acceptance_windows.ps1` tests `shell-init` in an isolated child PowerShell, proving
  that the project-local function shadows a conflicting global launcher only in that child.

## P19 — workflow visibility

- Added `/workflow`, a local read-only snapshot of optional plan state, latest change inventory,
  verification/repair readiness and the next safe user action. Plan approval remains advisory;
  all tool calls still flow through the normal permission policy.

## P20 — release gate

- Added `scripts/release_check_windows.ps1` for a project-local, no-cloud release check:
  package dependencies/fixtures, Windows session launcher, whitespace, lint, strict types and
  tests. It disables Python bytecode and pytest cache writes and never deletes files.

## Verification

The release gate is designed to run from the project `.venv`. The Docker security test remains
separate because it requires an explicitly provisioned local Docker image and
`REPOPILOT_RUN_DOCKER_SECURITY=1`.
