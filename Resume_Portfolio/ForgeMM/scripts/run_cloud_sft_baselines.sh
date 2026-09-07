#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_ENV="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-${HOME}/resume-project-assets/forgemm/models/Qwen2.5-VL-3B-Instruct}"
RUNS="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/cloud_stage04}"
DATA_DIR="${RUNS}/data"
STEPS="${FORGEMM_SFT_STEPS:-1000}"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS=1
cd "${ROOT}"

run_sft() {
  local name="$1"
  local dataset="$2"
  local output="${RUNS}/${name}"
  local exit_path="${RUNS}/${name}.exit"
  if [[ -f "${exit_path}" && "$(cat "${exit_path}")" == "0" ]] \
    && [[ -d "${output}/checkpoint-${STEPS}" ]]; then
    echo "skip completed ${name}"
    return 0
  fi

  set +e
  swift sft \
    --model "${MODEL}" \
    --dataset "${dataset}" \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --freeze_vit true \
    --freeze_aligner true \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 16 \
    --max_pixels 262144 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --max_steps "${STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --seed 42 \
    --data_seed 42 \
    --add_version false \
    --report_to none \
    --output_dir "${output}" \
    > "${output}.log" 2>&1
  local code=$?
  set -e
  echo "${code}" > "${exit_path}"
  return "${code}"
}

run_sft e1_answer_sft_1000 "${DATA_DIR}/answer_sft_train.jsonl"
run_sft e2_structured_sft_1000 "${DATA_DIR}/structured_sft_train.jsonl"
