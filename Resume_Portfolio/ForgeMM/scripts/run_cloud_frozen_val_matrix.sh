#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-/root/autodl-tmp/ForgeMM}"
PYTHON_ENV="${FORGEMM_ENV:-/root/autodl-tmp/envs/forgemm}"
MODEL="${FORGEMM_MODEL:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
DATASET="${FORGEMM_VAL_DATASET:-${ROOT}/artifacts/runs/cloud_stage04/data/chartqa_val_strict_eval.jsonl}"
RUNS="${ROOT}/artifacts/runs/cloud_stage04"
SAMPLES="$(wc -l < "${DATASET}")"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export FORGEMM_TRL_TRANSFORMERS_451=1
export OMP_NUM_THREADS=1
cd "${ROOT}"

run_eval() {
  local name="$1"
  local adapter="$2"
  local result="${RUNS}/frozen_val_${name}.jsonl"
  local infer_exit="${RUNS}/frozen_val_${name}.infer.exit"
  local metrics="${RUNS}/frozen_val_${name}.metrics.json"
  local metrics_exit="${RUNS}/frozen_val_${name}.metrics.exit"
  if [[ -f "${infer_exit}" && -f "${metrics_exit}" ]] \
    && [[ "$(cat "${infer_exit}")" == "0" ]] \
    && [[ "$(cat "${metrics_exit}")" == "0" ]]; then
    echo "skip completed ${name}"
    return 0
  fi

  local adapter_args=()
  if [[ "${adapter}" != "NONE" ]]; then
    adapter_args=(--adapters "${adapter}")
  fi

  set +e
  python "${PYTHON_ENV}/lib/python3.12/site-packages/swift/cli/infer.py" \
    --model "${MODEL}" \
    "${adapter_args[@]}" \
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
    > "${RUNS}/frozen_val_${name}.infer.log" 2>&1
  local code=$?
  echo "${code}" > "${infer_exit}"
  if [[ "${code}" -eq 0 ]]; then
    python scripts/evaluate_infer_results.py --input "${result}" --output "${metrics}" \
      > "${RUNS}/frozen_val_${name}.metrics.log" 2>&1
    code=$?
  fi
  echo "${code}" > "${metrics_exit}"
  set -e
  return "${code}"
}

run_eval e2_structured "${RUNS}/e2_structured_sft_1000/checkpoint-1000"
for seed in 17 42 2026; do
  run_eval "e3_task_seed${seed}" "${RUNS}/formal50_e3_task_seed${seed}/checkpoint-50"
  run_eval "e4_fixed_seed${seed}" "${RUNS}/formal50_e4_fixed_seed${seed}/checkpoint-50"
  run_eval "e5_dynamic_seed${seed}" "${RUNS}/formal50_e5_dynamic_seed${seed}/checkpoint-50"
done
run_eval a1_no_evidence_seed42 "${RUNS}/formal50_a1_no_evidence_seed42/checkpoint-50"
run_eval a2_no_operation_seed42 "${RUNS}/formal50_a2_no_operation_seed42/checkpoint-50"
