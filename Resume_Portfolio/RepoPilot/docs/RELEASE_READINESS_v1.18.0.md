# v1.18.0 release-readiness record

This record closes the P52–P60 engineering phase without claiming that an
unavailable host dependency was tested. It distinguishes implemented gates
from gates that remain blocked by the local machine.

## P52 — non-destructive release baseline

`scripts/release_baseline_windows.ps1` records Git HEAD, branch, porcelain
status counts and `git diff --check` into a retained artifact. It never stages,
commits, resets, deletes or repairs a working tree. The local P52 record shows
that the repository is intentionally not a clean commit baseline yet: release
integration must happen only after a human selects which existing changes and
reference materials belong in a commit.

## P53 — Docker safety gate

`scripts/docker_security_gate_windows.ps1 -BuildImage` builds the reviewed,
pinned sandbox image and runs only the gated live Docker security test. If the
Docker CLI or daemon is unavailable, it exits with a blocker and never falls
back to host execution. The current host has Docker Desktop service stopped and
the `dockerDesktopLinuxEngine` pipe unavailable, so the live test was not
claimed as passed. CI retains the Ubuntu live Docker job.

## P54–P57 — durable interactive operation

- A new integration test cancels a tool-running turn, closes the runtime,
  creates a fresh runtime and continues from the same stored checkpoint.
- DeepSeek now treats an SSE stream with chunks but no terminal marker as a
  recoverable provider failure; generic local OpenAI-compatible servers retain
  compatibility with streams that do not promise that marker.
- `/changes` provides a local-only compact view of the latest durable session
  revision, file-level additions/removals and the guarded `/diff`/`/rewind`
  path. Existing edit preview, explicit approval, conflict-aware undo and
  revision rewind remain the enforcement layer.
- Read-only acceptance source redaction now preserves Python expressions while
  masking literal credential values. This prevents a review package from being
  syntactically altered by an over-broad `secret = ...` rule.

## P58 — bounded real-project DeepSeek acceptance

Two no-tool, no-edit, read-only receipts were written for exactly these files:

1. `src/repopilot/providers/deepseek.py`
2. `src/repopilot/providers/base.py`
3. `tests/unit/test_deepseek_provider.py`

The first receipt, `p58_provider_contract_readonly`, is integrity-valid but
**not accepted as a semantic review**: the old source redactor rewrote ordinary
code expressions and caused a false syntax finding. After the source-redaction
fix, `p58_provider_contract_semantic_readonly` completed over SSE with a final
marker. Its suggestions were checked locally: the alleged Authorization test
failure is contradicted by the passing local test, response reasoning parsing
was outside that bounded source package, and the environment-name concern was
conditional. No external finding automatically produced a code change.

## P59–P60 — CI and release procedure

The GitHub Actions workflow now includes a Windows 3.12 project-local venv and
offline acceptance gate in addition to Ubuntu quality and live Docker jobs.
It has been configuration-reviewed locally but cannot be remotely executed
until a maintainer pushes the selected baseline. The normal release gate
remains offline with respect to model providers; optional gates are explicit:

```powershell
.\scripts\release_check_windows.ps1 -Baseline
.\scripts\release_check_windows.ps1 -PackageSmoke
.\scripts\release_check_windows.ps1 -DockerSecurity  # only after Docker is healthy
```

No Qwen credential, configuration or provider request was used in P52–P60.
