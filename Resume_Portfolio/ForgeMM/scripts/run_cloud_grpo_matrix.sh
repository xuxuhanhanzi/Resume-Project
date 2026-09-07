#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_ENV="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
MODEL="${FORGEMM_MODEL:-${HOME}/resume-project-assets/forgemm/models/Qwen2.5-VL-3B-Instruct}"
RUNS="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/cloud_stage04}"
ADAPTER="${FORGEMM_ADAPTER:-${RUNS}/e2_structured_sft_1000/checkpoint-1000}"
DATASET="${FORGEMM_GRPO_DATASET:-${RUNS}/data/grpo_train.jsonl}"

case "${MODE}" in
  quick)
    STEPS=20
    SEEDS=(42)
    ;;
  formal50)
    STEPS=50
    SEEDS=(17 42 2026)
    ;;
  *)
    echo "usage: $0 {quick|formal50}" >&2
    exit 2
    ;;
esac

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
export FORGEMM_TRL_TRANSFORMERS_451=1
export OMP_NUM_THREADS=1
cd "${ROOT}"

run_variant() {
  local variant="$1"
  local seed="$2"
  shift 2
  local output="${RUNS}/${MODE}_${variant}_seed${seed}"
  set +e
  swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --adapters "${ADAPTER}" \
    --dataset "${DATASET}" \
    --split_dataset_ratio 0 \
    --external_plugins "${ROOT}/src/forgemm/swift_plugin.py" \
    --tuner_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_generations 2 \
    --generation_batch_size 2 \
    --use_vllm false \
    --max_completion_length 256 \
    --max_steps "${STEPS}" \
    --learning_rate 1e-6 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps "${STEPS}" \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --add_version false \
    --report_to none \
    --output_dir "${output}" \
    "$@" \
    > "${output}.log" 2>&1
  local code=$?
  set -e
  echo "${code}" > "${output}.exit"
  return "${code}"
}

for seed in "${SEEDS[@]}"; do
  run_variant e3_task "${seed}" \
    --reward_funcs forgemm_task \
    --scale_rewards group

  run_variant e4_fixed "${seed}" \
    --reward_funcs forgemm_task forgemm_evidence forgemm_operation \
    --reward_weights 1 1 1 \
    --scale_rewards gdpo

  export FORGEMM_ENABLE_CHART_FGRPO=1
  export FORGEMM_TAU_EVIDENCE=0.90
  export FORGEMM_TAU_OPERATION=0.95
  export FORGEMM_DUAL_LR=0.01
  export FORGEMM_LAMBDA_MAX=5.0
  run_variant e5_dynamic "${seed}" \
    --reward_funcs forgemm_task forgemm_evidence forgemm_operation \
    --scale_rewards gdpo
  unset FORGEMM_ENABLE_CHART_FGRPO
done

if [[ "${MODE}" == "formal50" ]]; then
  export FORGEMM_ENABLE_CHART_FGRPO=1
  export FORGEMM_DUAL_LR=0.01
  export FORGEMM_LAMBDA_MAX=5.0

  export FORGEMM_TAU_EVIDENCE=0.0
  export FORGEMM_TAU_OPERATION=0.95
  run_variant a1_no_evidence 42 \
    --reward_funcs forgemm_task forgemm_evidence forgemm_operation \
    --scale_rewards gdpo

  export FORGEMM_TAU_EVIDENCE=0.90
  export FORGEMM_TAU_OPERATION=0.0
  run_variant a2_no_operation 42 \
    --reward_funcs forgemm_task forgemm_evidence forgemm_operation \
    --scale_rewards gdpo

  unset FORGEMM_ENABLE_CHART_FGRPO
fi
