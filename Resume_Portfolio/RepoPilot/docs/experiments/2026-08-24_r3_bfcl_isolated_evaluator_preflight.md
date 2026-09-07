# R3 BFCL v4：E 盘隔离评测器预检

## Status

Environment gate passed; no BFCL model score is reported.

## Isolated environment

- Upstream checkout: `external/BFCL/berkeley-function-call-leaderboard` at
  `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
- Interpreter: Python 3.10.6 virtual environment at
  `E:\RepoPilotEval\bfcl-venv`.
- The upstream package was installed editable with its declared dependencies.
  Its full command-line import initially failed because `qwen-agent` imports
  the undeclared dependency `soundfile`; `soundfile==0.14.0` was installed in
  this isolated environment only.  No RepoPilot dependency, source file, or
  BFCL source file was changed.
- Verified commands: `python -c "import bfcl_eval"`,
  `python -m bfcl_eval --help`, `generate --help`, and `evaluate --help`.
  The upstream CLI exposes both response generation and its AST/executable
  evaluator.
- The upstream `version` subcommand is not used as a receipt: it queries
  package metadata for `bfcl`, while this checkout installs the distribution as
  `bfcl_eval`, and therefore raises `PackageNotFoundError`.  The pinned Git
  commit above, not that broken display command, is the evaluator identity.

## Why this is not an R3 result

The local model used by the existing R3 smoke is `qwen2.5:1.5b`; it is not a
supported model identifier in this checkout's model registry (the registered
Qwen local entries are Qwen3 variants).  Binding an arbitrary Ollama model to
one of those handlers would change the upstream generation protocol, so it
cannot be described as an official BFCL generation.

Independently, the pre-registered R3-A treatment is stage-scoped tool exposure.
BFCL's single-turn calls contain no task-stage label.  Inferring a stage from a
gold call trajectory would leak the answer, while inventing a label after seeing
the prompts would amend the experiment.  Thus this environment gate enables a
future registered B0/B1 adapter, but it does not validate R3-A or replace the
existing correctly labelled local B0/B1 contract smoke.
