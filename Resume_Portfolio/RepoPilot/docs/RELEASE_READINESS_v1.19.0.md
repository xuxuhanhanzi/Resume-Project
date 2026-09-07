# v1.19.0 P61–P66 release-reliability record

This record distinguishes evidence that was executed locally from external
actions that require a maintainer-selected commit or Git remote. It does not
claim that a dirty working tree has been published.

## P61 — reviewable release scope

`scripts/release_scope_windows.ps1` writes a retained, non-destructive
classification of every Git porcelain entry. It labels product candidates,
local state, generated evidence and reference material, but never stages or
commits them. The latest inventory is at
`artifacts/release_scope/20260823_213908/release-scope.json` and contains 141
entries. A maintainer must decide the exact commit scope before publishing.

## P62 — Docker and CI evidence

Docker Desktop was started through its official local CLI and the explicit
container gate was run successfully:

```powershell
.\scripts\docker_security_gate_windows.ps1 -BuildImage
```

The pinned sandbox image built and the live security-boundary test passed
(`1 passed`). `scripts/ci_preflight_windows.ps1` verifies the workflow YAML,
Docker state and Git remote prerequisites while writing a retained report. The
current repository still has no `origin` remote, so GitHub Actions cannot be
triggered until a maintainer chooses a remote and pushes a reviewed commit.

## P63 — controlled provider interaction

`scripts/interactive_acceptance_windows.ps1` makes one explicit DeepSeek call
using only a fixed no-tool marker prompt. It does not select or intentionally
send project files, test output or user task text. The provider returned the
expected marker with zero tool calls, then deterministic tests covered
cancellation, cross-runtime recovery, edit revisions, undo and permissions.
Its retained receipt is in
`artifacts/interactive_acceptance/20260823_213112/`.

## P64 — offline Claude-style interaction transcript

`scripts/interactive_transcript_windows.ps1` drives `/status`, `/workflow`,
`/changes`, `/context` and `/exit` through a local buffered provider. It makes
no model request or tool call, and asserts that project/model/status/workflow
evidence appears in the saved transcript. The latest artifact is
`artifacts/interactive_transcript/20260823_213140/`.

## P65 — package identity

The opt-in package smoke now records the locally-built wheel name and SHA-256
in `package-smoke.receipt.json`; `verify_package_smoke_receipt_windows.ps1`
checks the retained wheel without installing or contacting a package index.
This is an integrity check for local release evidence, not a cryptographic
publisher signature or an auto-update channel.

## P66 — evidence boundaries and holdout gate

`repopilot eval coding-dev dashboard <run...>` aggregates only local synthetic
development receipts and marks legacy/unbound evidence plainly. It never calls
a provider and never reports a capability score. `repopilot eval coding-dev
holdout-plan` exposes only the metadata contract for the separate five-task
SWE-bench-Live official-evaluator release holdout; task identities, hidden
tests, gold patches and evaluator payloads are deliberately omitted from the
package. Synthetic results cannot replace that holdout. A fresh public
development diagnostic (`p66_development_20260823_01`) completed with all four
fixtures verified and a `bound_verified` outcome digest; it remains development
evidence rather than a benchmark result.

Qwen remains unconfigured and was not called during P61–P66.
