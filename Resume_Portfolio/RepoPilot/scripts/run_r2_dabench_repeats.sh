#!/usr/bin/env bash
# R2 — re-establish frozen DABench baselines.
# 3 repeats per config (base / guardrail / D3), full 35-question val35 set each.
#
# Preconditions (operator must satisfy before launch):
#   - Docker Desktop running (daemon reachable: `docker info` works)
#   - Ollama serving qwen2.5:7b (digest below) at http://127.0.0.1:11434
#   - project .venv activated with repopilot + docker + openai deps installed
#
# Each run writes artifacts/benchmarks/<run-id>/results.json (never overwrites;
# the runner refuses an existing run dir). Run-ids are unique per launch.
#
# NOTE: this script avoids `tee`/`date` (not always present in minimal MSYS
# shells) and logs via plain redirection to logs/r2_dabench_repeats.log.
set -u

cd "$(dirname "$0")/.." || exit 1

MODEL_REVISION="845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
MANIFEST="evaluation/benchmarks/manifests/dabench_validation35_v2.json"
PY="${PYTHON:-.venv/Scripts/python}"
LOG="logs/r2_dabench_repeats.log"

ts() { date -u +%FT%TZ 2>/dev/null || echo "T$(date +%s 2>/dev/null || echo ?)"; }

if [ ! -x "$PY" ] && [ ! -f "$PY" ]; then
  echo "ERROR: python not found at $PY (set PYTHON=... or create .venv)" >> "$LOG"
  exit 1
fi

mkdir -p logs

run_one() {
  local run_id="$1"; shift
  {
    echo "===== $run_id  $(ts) ====="
    "$PY" scripts/run_dabench_agent_smoke.py \
      --run-id "$run_id" \
      --model-revision "$MODEL_REVISION" \
      --manifest "$MANIFEST" \
      "$@"
    echo "----- $run_id exit=$? -----"
  } >> "$LOG" 2>&1
}

{
  echo "R2 batch start $(ts)"
  for i in 1 2 3; do
    run_one "r2_20260809_base_rep$i"     --no-dtype-guardrail
    run_one "r2_20260809_guardrail_rep$i"
    run_one "r2_20260809_d3_rep$i"       --parse-robustness
  done
  echo "R2 batch complete $(ts)"
} >> "$LOG" 2>&1
