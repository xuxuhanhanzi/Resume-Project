#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_DIR="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET="${FORGEMM_CONTROLLED_DATASET:-${ROOT}/datasets/ForgeMM-Controlled-v1-Lite}"
RUN_DIR="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/local_controlled_v1_lite}"
MAX_PIXELS="${FORGEMM_MAX_PIXELS:-200704}"

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS=1
cd "${ROOT}"

python scripts/audit_controlled_dataset.py \
  --dataset "${DATASET}" \
  --output "${RUN_DIR}/dataset_audit.json"
python scripts/local_lite_runtime_audit.py --output "${RUN_DIR}/runtime_audit.json"

mkdir -p "${RUN_DIR}"
set +e
swift sft \
  --model "${MODEL}" \
  --dataset "${DATASET}/views/sft_train.jsonl" \
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
  --max_pixels "${MAX_PIXELS}" \
  --max_length 1024 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --add_version false \
  --output_dir "${RUN_DIR}/sft_smoke" \
  > "${RUN_DIR}/sft_smoke.log" 2>&1
CODE=$?
set -e
echo "${CODE}" > "${RUN_DIR}/sft_smoke.exit"
exit "${CODE}"
