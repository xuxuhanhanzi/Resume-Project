# Experiment Record: Stage 1–6 implementation qualification

## Goal and hypothesis

Build a compact coding-agent technology lab that demonstrates the critical Agent mechanisms in
one inspectable runtime without claiming production completeness or model capability.

## Environment

- Date/timezone: 2026-07-28, Asia/Singapore
- OS: Windows
- Python qualification: 3.12.3
- Offline model: ScriptedProvider
- Local Qwen: adapter implemented; checkpoint/backend formal run not performed
- Docker: CLI present, daemon unavailable; no untrusted code executed

## Fixed commands

```powershell
..\ForgeLLM\.venv\Scripts\python.exe scripts\dev.py format
..\ForgeLLM\.venv\Scripts\python.exe scripts\dev.py check
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir artifacts\dist
python -m repopilot demo scripted --artifacts artifacts
python -m repopilot demo mcp --artifacts artifacts
python -m repopilot demo permission --artifacts artifacts
python -m repopilot demo recovery --artifacts artifacts
```

## Results

- Ruff format: 77 files formatted
- Ruff lint: passed
- strict mypy: 55 source/test/script files, no issues
- pytest: 33 passed
- Python micro benchmark: 10/10 Task Specs load, 10/10 begin fail-to-pass, 10/10 BM25 top-1
  localization identifies the permitted `module.py`
- scripted integrated run: Completed, 6 model iterations, 5 successful tools, deterministic
  verifier passed, forbidden `verify.py` absent from model-visible listing
- MCP demo: initialize, tools/list, and tools/call passed
- permission demo: `pyproject.toml` edit classified Require Approval
- recovery demo: pending action restored and completed action replayed from idempotency journal
- wheel: `repopilot-0.1.0.dev0-py3-none-any.whl`, SHA-256
  `d87d0d437a7b48471428d16196ab62dcf7ba3f3e0d5b086cfdb2149603eda9da`

## Failure found and corrected

The first real CLI demo used a relative workspace. The Patch write succeeded, but rendering its
relative result path failed, so the tool reported failure after a side effect. The deterministic
verifier still passed. Workspace normalization and transactional rollback were added, forbidden
paths were filtered from listing/search/retrieval, and regression tests were registered.

## Claim boundary and next decision

The result proves runtime mechanics on trusted micro fixtures. It does not prove Qwen coding
quality, full MCP compatibility, Docker isolation in this environment, hidden-test quality,
SWE-bench performance, or production readiness. Before teaching material is written, freeze and
run a local Qwen checkpoint on fresh copies of the ten tasks, then register model hash,
quantization, backend, prompt, tool schemas, budget, raw traces, and deterministic outcomes.
