# RepoPilot v1.17.0

## P44 — externally suggested issues are advisory until locally verified

`docs/P40_ADVISORY_TRIAGE.md` records each P40 read-only assessment point,
its local evidence and disposition. The external response was not treated as a
code change or proof. The claimed symlink issue was not accepted without a
reproducer; local path resolution already rejects symlink escapes. No model or
Qwen request was made in this release work.

## P45–P47 — cancellation, process, Docker and text-boundary hardening

- `CancellationToken` now supports cancellation initiated by a service/UI
  thread and wakes the owning asyncio loop safely. If a process completion and
  cancellation arrive together, cancellation takes precedence.
- POSIX local commands start in their own session; cancellation terminates the
  owned process group. Windows retains its explicit `taskkill /T` tree path.
- The Docker workspace bind mount is explicitly `readonly`; the gated live
  security probe now verifies that workspace writes fail. Container root was
  already read-only, but a bind mount needs this separate flag.
- Bounded output now truncates on a UTF-8 code-point boundary rather than
  rendering an artificial replacement glyph.

## P48–P49 — diagnosable empty responses and durable recovery

An empty provider final response is now a classified model failure, not a
misleading completed turn. The recovery message distinguishes output-budget,
content-filter, incomplete-stream and reasoning-only cases without exposing
raw request/response content. Cancellation regression coverage now proves a
durably cancelled session can continue in a later turn.

## P50–P51 — Windows operations and installation evidence

- `docs/WINDOWS_ENVIRONMENT_RECOVERY.md` gives a manual, non-destructive path
  for Anaconda/PATH collisions. It explicitly avoids editing PATH, `$PROFILE`,
  Credential Manager and old package metadata.
- `scripts/package_smoke_windows.ps1` is an opt-in clean-virtual-environment
  package smoke check. It builds a wheelhouse, installs it using only that
  wheelhouse, checks the console launcher and `doctor`, then retains artifacts
  for inspection. `scripts/release_check_windows.ps1 -PackageSmoke` invokes it
  explicitly; the ordinary offline release gate does not resolve packages.
