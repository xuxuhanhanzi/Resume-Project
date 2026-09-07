#!/usr/bin/env bash
set -euo pipefail

# Paired answer-only vs visual-structured SFT.  The run directory is immutable:
# a second attempt must use a distinct FORGEMM_RUN_ID instead of overwriting logs.
ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_DIR="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET="${FORGEMM_CONTROLLED_DATASET:-${ROOT}/datasets/ForgeMM-Controlled-v1-Lite}"
RUN_ID="${FORGEMM_RUN_ID:-e0_sft_1000}"
RUN_DIR="${ROOT}/artifacts/runs/local_controlled_v1_lite/${RUN_ID}"
STEPS="${FORGEMM_SFT_STEPS:-1000}"
MAX_PIXELS="${FORGEMM_MAX_PIXELS:-200704}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing_existing_run:${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"
source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS=1
cd "${ROOT}"

python scripts/audit_controlled_dataset.py \
  --dataset "${DATASET}" --output "${RUN_DIR}/dataset_audit.json"
python scripts/local_lite_runtime_audit.py --output "${RUN_DIR}/runtime_audit.json"

run_sft() {
  local name="$1"
  local dataset="$2"
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
    --max_pixels "${MAX_PIXELS}" \
    --max_length 1024 \
    --max_steps "${STEPS}" \
    --save_strategy steps \
    --save_steps "${STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --seed 42 \
    --data_seed 42 \
    --report_to none \
    --add_version false \
    --output_dir "${RUN_DIR}/${name}" \
    > "${RUN_DIR}/${name}.log" 2>&1
}

run_infer() {
  local name="$1"
  local adapter="$2"
  swift infer \
    --model "${MODEL}" \
    --adapters "${adapter}" \
    --infer_backend transformers \
    --val_dataset "${DATASET}/views/val_prompts.jsonl" \
    --val_dataset_sample 600 \
    --max_new_tokens 160 \
    --max_batch_size 1 \
    --temperature 0 \
    --stream false \
    --attn_impl sdpa \
    --max_pixels "${MAX_PIXELS}" \
    --quant_method bnb \
    --quant_bits 4 \
    --result_path "${RUN_DIR}/val_${name}.jsonl" \
    > "${RUN_DIR}/val_${name}.infer.log" 2>&1
  python scripts/evaluate_controlled_infer_results.py \
    --input "${RUN_DIR}/val_${name}.jsonl" \
    --oracle "${DATASET}/manifests/val_oracle.jsonl" \
    --output "${RUN_DIR}/val_${name}.metrics.json" \
    > "${RUN_DIR}/val_${name}.metrics.log" 2>&1
}

run_sft answer_sft "${DATASET}/views/answer_sft_train.jsonl"
run_sft structured_sft "${DATASET}/views/sft_train.jsonl"
run_infer answer "${RUN_DIR}/answer_sft/checkpoint-${STEPS}"
run_infer structured "${RUN_DIR}/structured_sft/checkpoint-${STEPS}"
python scripts/check_controlled_sft_gate.py \
  --baseline "${RUN_DIR}/val_answer.jsonl" \
  --structured "${RUN_DIR}/val_structured.jsonl" \
  --oracle "${DATASET}/manifests/val_oracle.jsonl" \
  --output "${RUN_DIR}/structured_sft_gate.json" \
  > "${RUN_DIR}/structured_sft_gate.log" 2>&1
