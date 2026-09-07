#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${EXPERIMENT_ENVS_ROOT:-${HOME}/.venvs}"
ENV_DIR="${REPOPILOT_ENV:-${ENV_ROOT}/repopilot}"
PYTHON="${ENV_DIR}/bin/python"
RUN_ID="${EXPERIMENT_RUN_ID:-${PROFILE}-latest}"
SAFE_ID="${RUN_ID//[^A-Za-z0-9_.-]/_}"
MODEL="${REPOPILOT_MODEL:-qwen2.5:7b}"
MODEL_REVISION="${REPOPILOT_MODEL_REVISION:-$(curl -fsS http://127.0.0.1:11434/api/tags | \
  "${PYTHON}" -c 'import json,sys; name=sys.argv[1]; rows=json.load(sys.stdin).get("models", []); print(next((row["digest"] for row in rows if row.get("name")==name), ""))' "${MODEL}")}"
FRAMES_COUNT=3
STATE_DIR="${ROOT}/artifacts/experiment_pc/${SAFE_ID}/steps"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
[[ -x "${PYTHON}" ]] || { echo "RepoPilot environment missing; run setup first" >&2; exit 2; }
[[ -n "${MODEL_REVISION}" ]] || { echo "Could not resolve Ollama model digest for ${MODEL}" >&2; exit 2; }
mkdir -p "${STATE_DIR}"

run_step() {
  local name="$1"
  shift
  local marker="${STATE_DIR}/${name}.exit"
  if [[ -f "${marker}" && "$(cat "${marker}")" == "0" ]]; then
    echo "skip completed ${name}"
    return 0
  fi
  set +e
  "$@"
  local code=$?
  set -e
  echo "${code}" > "${marker}"
  return "${code}"
}

cd "${ROOT}"
run_step quality "${PYTHON}" scripts/dev.py
run_step capacity "${PYTHON}" scripts/benchmark_local_model_capacity.py \
  --output "artifacts/benchmarks/${SAFE_ID}_capacity.json" \
  --model "${MODEL}" --model-revision "${MODEL_REVISION}" --tasks 12

if [[ "${PROFILE}" == "full" ]]; then FRAMES_COUNT=60; fi
for repetition in 1 2 3; do
  run_step "frames_rep${repetition}" "${PYTHON}" scripts/run_frames_oracle_smoke.py \
    --run-id "${SAFE_ID}_frames${FRAMES_COUNT}_rep${repetition}" --count "${FRAMES_COUNT}" \
    --model "${MODEL}" --model-revision "${MODEL_REVISION}" --retriever-type bm25
done

MANIFEST_ARGS=()
if [[ "${PROFILE}" == "full" ]]; then
  MANIFEST_ARGS=(--manifest evaluation/benchmarks/manifests/dabench_validation35_v2.json)
fi
for repetition in 1 2 3; do
  run_step "dabench_base_rep${repetition}" "${PYTHON}" scripts/run_dabench_agent_smoke.py \
    --run-id "${SAFE_ID}_dabench_base_rep${repetition}" --model "${MODEL}" \
    --model-revision "${MODEL_REVISION}" "${MANIFEST_ARGS[@]}" --no-dtype-guardrail
  run_step "dabench_guardrail_rep${repetition}" "${PYTHON}" scripts/run_dabench_agent_smoke.py \
    --run-id "${SAFE_ID}_dabench_guardrail_rep${repetition}" --model "${MODEL}" \
    --model-revision "${MODEL_REVISION}" "${MANIFEST_ARGS[@]}"
  run_step "dabench_robust_rep${repetition}" "${PYTHON}" scripts/run_dabench_agent_smoke.py \
    --run-id "${SAFE_ID}_dabench_robust_rep${repetition}" --model "${MODEL}" \
    --model-revision "${MODEL_REVISION}" "${MANIFEST_ARGS[@]}" --parse-robustness
done

if [[ "${PROFILE}" == "full" ]]; then
  REVISION="a637bd46829f3132e12938c8a0ca93173a977b8e"
  DATA_ROOT="evaluation/benchmarks/data/swebench_live/${REVISION}"
  run_step swebench_generation "${PYTHON}" scripts/run_swebench_live_fresh_holdout.py \
    --run-id "${SAFE_ID}_swebench_fresh5" \
    --public-jsonl "${DATA_ROOT}/fresh5/fresh_public.jsonl" \
    --task-config "${DATA_ROOT}/fresh5/fresh_task_config.json" \
    --model "${MODEL}" --model-revision "${MODEL_REVISION}"
  run_step swebench_evaluation "${PYTHON}" scripts/eval_swebench_resolved.py \
    --predictions "artifacts/benchmarks/${SAFE_ID}_swebench_fresh5/predictions.jsonl" \
    --parquet "${DATA_ROOT}/lite.parquet" \
    --workspace-root evaluation/benchmarks/workspaces/swebench_live
fi
