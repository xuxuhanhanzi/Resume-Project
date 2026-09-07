#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_ENV="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-${HOME}/resume-project-assets/forgemm/models/Qwen2.5-VL-3B-Instruct}"
RUNS="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/cloud_stage04}"
SUMMARY="${RUNS}/frozen_val_summary.json"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export FORGEMM_TRL_TRANSFORMERS_451=1
export OMP_NUM_THREADS=1
cd "${ROOT}"

selected_seed() {
  local method="$1"
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoints"][sys.argv[2]]["seed"])' "${SUMMARY}" "${method}"
}

run_task_eval() {
  local dataset_name="$1"
  local dataset="$2"
  local method="$3"
  local adapter="$4"
  local samples
  samples="$(wc -l < "${dataset}")"
  local name="${dataset_name}_${method}"
  local raw="${RUNS}/frozen_task_${name}.raw.jsonl"
  local enriched="${RUNS}/frozen_task_${name}.jsonl"
  local infer_exit="${RUNS}/frozen_task_${name}.infer.exit"
  local metrics="${RUNS}/frozen_task_${name}.metrics.json"
  local metrics_exit="${RUNS}/frozen_task_${name}.metrics.exit"
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
    --result_path "${raw}" \
    > "${RUNS}/frozen_task_${name}.infer.log" 2>&1
  local code=$?
  echo "${code}" > "${infer_exit}"
  if [[ "${code}" -eq 0 ]]; then
    python scripts/merge_infer_metadata.py \
      --dataset "${dataset}" --inference "${raw}" --output "${enriched}" \
      > "${RUNS}/frozen_task_${name}.merge.log" 2>&1
    code=$?
  fi
  if [[ "${code}" -eq 0 ]]; then
    python scripts/evaluate_task_infer_results.py --input "${enriched}" --output "${metrics}" \
      > "${RUNS}/frozen_task_${name}.metrics.log" 2>&1
    code=$?
  fi
  echo "${code}" > "${metrics_exit}"
  set -e
  return "${code}"
}

e3_seed="$(selected_seed e3_task)"
e5_seed="$(selected_seed e5_dynamic)"
e3_adapter="${RUNS}/formal50_e3_task_seed${e3_seed}/checkpoint-50"
e5_adapter="${RUNS}/formal50_e5_dynamic_seed${e5_seed}/checkpoint-50"

for dataset_name in chartqa_test chartqapro_test; do
  dataset="${RUNS}/data/frozen_eval/${dataset_name}_full.jsonl"
  run_task_eval "${dataset_name}" "${dataset}" "e3_task_seed${e3_seed}" "${e3_adapter}"
  run_task_eval "${dataset_name}" "${dataset}" "e5_dynamic_seed${e5_seed}" "${e5_adapter}"
done
