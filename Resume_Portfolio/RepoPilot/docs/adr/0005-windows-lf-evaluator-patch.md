# ADR 0005: Windows LF Patch for SWE-bench-Live Evaluator

## Status

Accepted — 2026-08-07

## Context

The official SWE-bench-Live evaluator (`external/SWE-bench-Live-python-only`,
commit `ad79b850f15e33992e96f03f6e97f05ddf9aa0be`, branch `python-only`) writes
two files during evaluation:

1. `patch.diff` — the model prediction copied into the container.
2. `eval.sh` — the evaluation script copied into the container.

On Windows, `Path.write_text(content)` uses the platform default newline (`\r\n`).
When these CRLF-terminated files are copied into a Linux Docker container, `git apply`
and `bash eval.sh` can fail or produce silently different results because the extra
`\r` characters corrupt patch headers and shell syntax.

## Decision

Apply a local portability patch to `swebench/harness/run_evaluation.py` that forces
`newline="\n"` on both `write_text` calls:

```python
patch_file.write_text(pred[KEY_PREDICTION] or "", newline="\n")
eval_file.write_text(test_spec.eval_script, newline="\n")
```

This is the only modification to the evaluator submodule. It must not be removed
unless the project migrates to a Linux host and re-verifies that the official
evaluator produces identical results without the patch.

## Consequences

- The evaluator submodule carries one local diff that is not upstream.
- Any `git -C external/SWE-bench-Live-python-only checkout` or `pull` that
  overwrites `run_evaluation.py` will silently drop this patch; re-apply it
  from this ADR if that happens.
- The patch is purely a line-ending fix; it does not alter evaluation logic,
  scoring, or test selection.
- Windows CRLF issues are also the reason submission names are normalized
  (`qwen2.5_7b` instead of `qwen2.5:7b`) — the colon is illegal in Windows
  directory names used by the evaluator log path.
