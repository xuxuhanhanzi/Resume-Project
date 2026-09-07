#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_ENV="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
RUNS="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/cloud_stage04}"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
cd "${ROOT}"
mkdir -p "${RUNS}"

run_driver() {
  local name="$1"
  shift
  local exit_path="${RUNS}/${name}.exit"
  if [[ -f "${exit_path}" && "$(cat "${exit_path}")" == "0" ]]; then
    echo "skip completed ${name}"
    return 0
  fi
  set +e
  "$@" > "${RUNS}/${name}.log" 2>&1
  local code=$?
  set -e
  echo "${code}" > "${exit_path}"
  return "${code}"
}

run_driver prepare_data bash scripts/prepare_cloud_stage04_data.sh
run_driver stage04_data_gate python scripts/check_stage04_data_gate.py \
  --train-summary "${RUNS}/data/structured_training_summary.json" \
  --val-summary "${RUNS}/data/chartqa_val_strict_eval_summary.json" \
  --test-summary "${RUNS}/data/chartqa_test_strict_eval_summary.json" \
  --output "${RUNS}/stage04_data_gate.json"
run_driver sft_baselines bash scripts/run_cloud_sft_baselines.sh
run_driver frozen_baseline_val_driver bash scripts/run_cloud_baseline_eval.sh val
run_driver e2_gate python scripts/check_structured_sft_gate.py \
  --baseline "${RUNS}/frozen_val_e1_answer.jsonl" \
  --structured "${RUNS}/frozen_val_e2_structured.jsonl" \
  --output "${RUNS}/e2_gate.json"
run_driver formal_matrix_driver bash scripts/run_cloud_grpo_matrix.sh formal50
run_driver frozen_val_matrix_driver bash scripts/run_cloud_frozen_val_matrix.sh
run_driver frozen_val_summary python scripts/summarize_frozen_val_matrix.py \
  --run-dir "${RUNS}" --output "${RUNS}/frozen_val_summary.json"
run_driver frozen_baseline_driver bash scripts/run_cloud_baseline_eval.sh full
run_driver frozen_test_strict_driver bash scripts/run_cloud_selected_strict_test.sh
run_driver frozen_task_driver bash scripts/run_cloud_selected_task_tests.sh
run_driver formal_training_summary python scripts/summarize_cloud_training.py \
  --run-dir "${RUNS}" --output "${RUNS}/formal_training_summary.json"
run_driver final_summary python scripts/summarize_cloud_final.py \
  --run-dir "${RUNS}" --output "${RUNS}/cloud_final_summary.json"
