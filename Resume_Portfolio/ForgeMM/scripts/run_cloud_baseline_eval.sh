#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-/root/autodl-tmp/ForgeMM}"
PYTHON_ENV="${FORGEMM_ENV:-/root/autodl-tmp/envs/forgemm}"
MODEL="${FORGEMM_MODEL:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
RUNS="${ROOT}/artifacts/runs/cloud_stage04"
VAL_DATASET="${RUNS}/data/chartqa_val_strict_eval.jsonl"
TEST_DATASET="${RUNS}/data/chartqa_test_strict_eval.jsonl"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export FORGEMM_TRL_TRANSFORMERS_451=1
export OMP_NUM_THREADS=1
cd "${ROOT}"

run_eval() {
  local split="$1"
  local name="$2"
  local dataset="$3"
  local adapter="$4"
  local samples
  samples="$(wc -l < "${dataset}")"
  local result="${RUNS}/frozen_${split}_${name}.jsonl"
  local infer_exit="${RUNS}/frozen_${split}_${name}.infer.exit"
  local metrics="${RUNS}/frozen_${split}_${name}.metrics.json"
  local metrics_exit="${RUNS}/frozen_${split}_${name}.metrics.exit"
  if [[ -f "${infer_exit}" && -f "${metrics_exit}" ]] \
    && [[ "$(cat "${infer_exit}")" == "0" ]] \
    && [[ "$(cat "${metrics_exit}")" == "0" ]]; then
    echo "skip completed ${split}/${name}"
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
    --val_dataset "${dataset}" \
    --val_dataset_sample "${samples}" \
    --max_new_tokens 512 \
    --max_batch_size 4 \
    --temperature 0 \
    --stream false \
    --attn_impl sdpa \
    --max_pixels 262144 \
    --quant_method bnb \
    --quant_bits 4 \
    --result_path "${result}" \
    > "${RUNS}/frozen_${split}_${name}.infer.log" 2>&1
  local code=$?
  echo "${code}" > "${infer_exit}"
  if [[ "${code}" -eq 0 ]]; then
    python scripts/evaluate_infer_results.py --input "${result}" --output "${metrics}" \
      > "${RUNS}/frozen_${split}_${name}.metrics.log" 2>&1
    code=$?
  fi
  echo "${code}" > "${metrics_exit}"
  set -e
  return "${code}"
}

run_eval val e0_base "${VAL_DATASET}" NONE
run_eval val e1_answer "${VAL_DATASET}" "${RUNS}/e1_answer_sft_1000/checkpoint-1000"

run_eval test_strict e0_base "${TEST_DATASET}" NONE
run_eval test_strict e1_answer "${TEST_DATASET}" "${RUNS}/e1_answer_sft_1000/checkpoint-1000"
run_eval test_strict e2_structured "${TEST_DATASET}" \
  "${RUNS}/e2_structured_sft_1000/checkpoint-1000"
