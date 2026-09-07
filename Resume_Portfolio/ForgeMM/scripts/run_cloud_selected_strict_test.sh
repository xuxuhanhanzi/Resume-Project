#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_ENV="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-${HOME}/resume-project-assets/forgemm/models/Qwen2.5-VL-3B-Instruct}"
RUNS="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/cloud_stage04}"
DATASET="${FORGEMM_TEST_DATASET:-${RUNS}/data/chartqa_test_strict_eval.jsonl}"
SUMMARY="${RUNS}/frozen_val_summary.json"
SAMPLES="$(wc -l < "${DATASET}")"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export FORGEMM_TRL_TRANSFORMERS_451=1
export OMP_NUM_THREADS=1
cd "${ROOT}"

selected_seed() {
  local method="$1"
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoints"][sys.argv[2]]["seed"])' "${SUMMARY}" "${method}"
}

run_eval() {
  local name="$1"
  local adapter="$2"
  local result="${RUNS}/frozen_test_strict_${name}.jsonl"
  local infer_exit="${RUNS}/frozen_test_strict_${name}.infer.exit"
  local metrics="${RUNS}/frozen_test_strict_${name}.metrics.json"
  local metrics_exit="${RUNS}/frozen_test_strict_${name}.metrics.exit"
  if [[ -f "${infer_exit}" && -f "${metrics_exit}" ]] \
    && [[ "$(cat "${infer_exit}")" == "0" ]] \
    && [[ "$(cat "${metrics_exit}")" == "0" ]]; then
    echo "skip completed ${name}"
    return 0
  fi
  set +e
  swift infer \
    --model "${MODEL}" \
    --adapters "${adapter}" \
    --infer_backend transformers \
    --val_dataset "${DATASET}" \
    --val_dataset_sample "${SAMPLES}" \
    --max_new_tokens 512 \
    --max_batch_size 4 \
    --temperature 0 \
    --stream false \
    --attn_impl sdpa \
    --max_pixels 262144 \
    --quant_method bnb \
    --quant_bits 4 \
    --result_path "${result}" \
    > "${RUNS}/frozen_test_strict_${name}.infer.log" 2>&1
  local code=$?
  echo "${code}" > "${infer_exit}"
  if [[ "${code}" -eq 0 ]]; then
    python scripts/evaluate_infer_results.py --input "${result}" --output "${metrics}" \
      > "${RUNS}/frozen_test_strict_${name}.metrics.log" 2>&1
    code=$?
  fi
  echo "${code}" > "${metrics_exit}"
  set -e
  return "${code}"
}

e3_seed="$(selected_seed e3_task)"
e4_seed="$(selected_seed e4_fixed)"
e5_seed="$(selected_seed e5_dynamic)"
run_eval "e3_task_seed${e3_seed}" "${RUNS}/formal50_e3_task_seed${e3_seed}/checkpoint-50"
run_eval "e4_fixed_seed${e4_seed}" "${RUNS}/formal50_e4_fixed_seed${e4_seed}/checkpoint-50"
run_eval "e5_dynamic_seed${e5_seed}" "${RUNS}/formal50_e5_dynamic_seed${e5_seed}/checkpoint-50"

python scripts/compare_structured_infer_results.py \
  --baseline "${RUNS}/frozen_test_strict_e3_task_seed${e3_seed}.jsonl" \
  --candidate "${RUNS}/frozen_test_strict_e5_dynamic_seed${e5_seed}.jsonl" \
  --output "${RUNS}/frozen_test_strict_e3_vs_e5.json"
