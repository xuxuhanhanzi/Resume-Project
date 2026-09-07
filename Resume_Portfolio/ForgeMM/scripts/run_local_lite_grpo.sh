#!/usr/bin/env bash
set -euo pipefail

# Precondition: a separate E0 run has written decision=advance in its validation gate.
ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_DIR="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET="${FORGEMM_CONTROLLED_DATASET:-${ROOT}/datasets/ForgeMM-Controlled-v1-Lite}"
E0_RUN="${FORGEMM_E0_RUN:?set FORGEMM_E0_RUN to the completed E0 run directory}"
RUN_ID="${FORGEMM_RUN_ID:-grpo_quick}"
RUN_DIR="${ROOT}/artifacts/runs/local_controlled_v1_lite/${RUN_ID}"
MODE="${1:-quick}"
MAX_PIXELS="${FORGEMM_MAX_PIXELS:-200704}"

case "${MODE}" in
  quick) STEPS=20; SEEDS=(42) ;;
  formal) STEPS=500; SEEDS=(17 42 2026) ;;
  *) echo "usage: $0 {quick|formal}" >&2; exit 2 ;;
esac

if [[ ! -f "${E0_RUN}/structured_sft_gate.json" ]] \
  || ! grep -q '"decision": "advance"' "${E0_RUN}/structured_sft_gate.json"; then
  echo "missing_or_failed_structured_sft_gate:${E0_RUN}" >&2
  exit 3
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing_existing_run:${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"
source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export FORGEMM_TRL_TRANSFORMERS_451=1
export OMP_NUM_THREADS=1
cd "${ROOT}"

python scripts/audit_controlled_dataset.py \
  --dataset "${DATASET}" --output "${RUN_DIR}/dataset_audit.json"
python scripts/local_lite_runtime_audit.py --output "${RUN_DIR}/runtime_audit.json"

run_variant() {
  local variant="$1"
  local seed="$2"
  shift 2
  swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --adapters "${E0_RUN}/structured_sft/checkpoint-${FORGEMM_SFT_STEPS:-1000}" \
    --dataset "${DATASET}/views/grpo_train.jsonl" \
    --split_dataset_ratio 0 \
    --external_plugins "${ROOT}/src/forgemm/swift_plugin.py" \
    --tuner_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --max_pixels "${MAX_PIXELS}" \
    --max_length 1024 \
    --num_generations 2 \
    --generation_batch_size 1 \
    --use_vllm false \
    --max_completion_length 160 \
    --max_steps "${STEPS}" \
    --learning_rate 1e-6 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps "${STEPS}" \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --add_version false \
    --report_to none \
    --output_dir "${RUN_DIR}/${variant}_seed${seed}" \
    "$@" \
    > "${RUN_DIR}/${variant}_seed${seed}.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  run_variant task "${seed}" \
    --reward_funcs forgemm_controlled_task \
    --scale_rewards group
  run_variant fixed "${seed}" \
    --reward_funcs forgemm_controlled_task forgemm_controlled_evidence forgemm_controlled_operation \
    --reward_weights 1 1 1 \
    --scale_rewards gdpo
  export FORGEMM_ENABLE_CHART_FGRPO=1
  export FORGEMM_TAU_EVIDENCE=0.90
  export FORGEMM_TAU_OPERATION=0.95
  export FORGEMM_DUAL_LR=0.01
  export FORGEMM_LAMBDA_MAX=5.0
  run_variant dynamic "${seed}" \
    --reward_funcs forgemm_controlled_task forgemm_controlled_evidence forgemm_controlled_operation \
    --scale_rewards gdpo
  unset FORGEMM_ENABLE_CHART_FGRPO
done
