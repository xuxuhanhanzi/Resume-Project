# ADR 0006: Windows LF Patch for the SWE-bench Verified Evaluator

## Status

Accepted — 2026-08-24

## Context

The isolated official evaluator checkout at `E:\RepoPilotEval\SWE-bench` is
SWE-bench commit `7a21e05772954cc81471ae19d56f436cecf43c54`.  On Windows,
Python's default `Path.write_text()` newline is CRLF.  The evaluator copies both
the model patch and an `eval.sh` script into a Linux Docker container, where CRLF
can corrupt a shell command or a patch header.

The unmodified evaluator was first exercised with its `--gold` patch for
`sympy__sympy-20590`.  The patch applied, but `eval.sh` failed before the test
environment activated (`set: pipefail\\r: invalid option name`).  This run is
preserved as an infrastructure diagnostic and is not an agent result.

## Decision

Apply only the following portability changes to
`swebench/harness/run_evaluation.py`:

```python
patch_file.write_text(pred["model_patch"] or "", newline="\\n")
eval_file.write_text(
    _inject_asset_restore(test_spec.eval_script, restore_cmds), newline="\\n"
)
```

The patched file SHA-256 is
`71450343B39D2B733A4254F63B7A027BE5E301CBA9E8E1772436626D34762C4F`.
No test selection, prediction, container image, asset-restore command, or scoring
logic is changed.

## Evidence and consequences

- Re-running the same official gold command after the patch resolved `1/1`
  (`sympy__sympy-20590`); the result is stored outside the repository at
  `E:\RepoPilotEval\runs\r2_20260824_official_gold_smoke_lf`.
- The gold patch was visible to the evaluator, so that instance is excluded from
  every agent-generation comparison.
- The evaluator checkout deliberately has this two-call local diff.  A future
  update or a Linux migration must re-check the file hash and reproduce the
  gold smoke before any new score is interpreted.
