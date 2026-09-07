# RepoPilot v1.13.0

## P21 — frozen public synthetic DeepSeek diagnostic

The explicitly authorized DeepSeek run used `deepseek/deepseek-v4-flash`, the bundled public
`repopilot-synthetic-dev-v1` suite and its four visible-test fixtures:

- Run ID: `deepseek_p21_20260823_01`
- Suite digest: `89b8e296250d01be1894baaabc52476770855d06a6532ea921211b5d52429cbb`
- Outcome: 3/4 verified, with localization, non-empty patch and patch application recorded for all four cases.
- The remaining `parse-boolean-token` case successfully applied a patch and ran its test, then the
  model requested optional `git_diff`; the 12,000 token task limit stopped the run before final
  deterministic verification. This is a development diagnostic, not a capability benchmark.

The corresponding local-only receipt and funnel records are under
`artifacts/coding_development/runs/deepseek_p21_20260823_01/`.

## P22–P23 — one bounded evidence-led change and comparison

P22 added an opt-in runtime path used only for `trusted_fixture=True` synthetic development tasks:
after a successful exact `apply_patch` and successful immutable `run_tests`, RepoPilot runs its
deterministic verifier immediately. It does not apply to normal interactive sessions or arbitrary
projects, and it does not broaden permissions or tool access.

P23 repeated the same public suite, provider, model, selected cases, tool surface and 12,000-token
task budget:

- Run ID: `deepseek_p23_autofinalize_20260823_01`
- Outcome: 4/4 verified.
- Its receipt explicitly records the runtime profile including
  `auto_finalize_after_successful_test=true`.

This is evidence that the observed post-test stop was removed in this small synthetic setup. It is
not a general coding-agent score, a holdout result, or proof of improvement on SWE-bench-Live.

## P24–P26 — intentionally retained consent boundaries

- Qwen remains unconfigured and uncalled, as requested earlier. A future comparison must use the
  same public suite/protocol and separate provider-specific receipts.
- `shell-doctor` continues to report the Anaconda PATH collision and stale pip metadata without
  modifying PATH, `$PROFILE`, Credential Manager, Anaconda, or existing package metadata.
- A real-project cloud acceptance is not automatically run: reading a repository file would send
  the selected content to DeepSeek. That requires a distinct approval naming both the project
  scope and the provider. Local scripted/integration coverage is still included in the release gate.

## P27 — release handoff

Run the no-cloud release procedure from the project root:

```powershell
.\scripts\release_check_windows.ps1
```

It checks the project-local launcher and package fixture, dependency consistency, diff whitespace,
Ruff, strict mypy and pytest. It does not create a remote release, tag, mutate global PATH, or use
provider credentials. The Docker security test remains explicitly gated by
`REPOPILOT_RUN_DOCKER_SECURITY=1` and a provisioned local image.

The final project-local validation for this release recorded: Ruff passing, strict mypy passing for
194 source files, `253 passed, 1 skipped` from pytest, `pip check` with no broken requirements and
`git diff --check` passing. The skipped test is the intentionally opt-in live Docker security gate.
