#!/usr/bin/env bash
set -euo pipefail

: "${AUTOVLA_ROOT:?Set AUTOVLA_ROOT to the AutoVLA checkout}"
: "${DRIVEVLA_GUARD_ROOT:?Set DRIVEVLA_GUARD_ROOT to this project root}"
: "${AUTOVLA_CHECKPOINT:?Set AUTOVLA_CHECKPOINT to the released .ckpt}"
: "${QWEN_MODEL_ROOT:?Set QWEN_MODEL_ROOT to the local Qwen2.5-VL-3B model directory}"
: "${AUTOVLA_MODEL_CONFIG:?Set AUTOVLA_MODEL_CONFIG to the upstream nuPlan YAML}"
: "${NAVSIM_JSON_DATA:?Set NAVSIM_JSON_DATA to preprocessed AutoVLA JSON data}"
: "${NAVSIM_SENSOR_DATA:?Set NAVSIM_SENSOR_DATA to NAVSIM sensor blobs}"
: "${NAVSIM_METRIC_CACHE:?Set NAVSIM_METRIC_CACHE to NAVSIM metric cache}"
: "${RUN_VARIANT:?Set RUN_VARIANT to b0, e2, e3, or e4}"

case "${RUN_VARIANT}" in
  b0) GUARD_CONFIG="configs/b0_fast_greedy.yaml" ;;
  e2) GUARD_CONFIG="configs/e2_rerank.yaml" ;;
  e3) GUARD_CONFIG="configs/e3_router.yaml" ;;
  e4) GUARD_CONFIG="configs/e4_guard.yaml" ;;
  *) echo "RUN_VARIANT must be one of: b0, e2, e3, e4" >&2; exit 2 ;;
esac

PREFLIGHT_OUTPUT="${DRIVEVLA_GUARD_ROOT}/artifacts/official/preflight_${RUN_VARIANT}.json"
python -m drivevla_guard.cli official-preflight \
  --autovla-root "${AUTOVLA_ROOT}" \
  --checkpoint "${AUTOVLA_CHECKPOINT}" \
  --qwen-model "${QWEN_MODEL_ROOT}" \
  --model-config "${AUTOVLA_MODEL_CONFIG}" \
  --json-data "${NAVSIM_JSON_DATA}" \
  --sensor-data "${NAVSIM_SENSOR_DATA}" \
  --metric-cache "${NAVSIM_METRIC_CACHE}" \
  --expected-upstream-commit ba34eed74ce6729e7986592d0e66cbaca397b4fa \
  --min-vram-gb "${MIN_VRAM_GB:-24}" \
  --output "${PREFLIGHT_OUTPUT}"

export PYTHONPATH="${DRIVEVLA_GUARD_ROOT}/src:${AUTOVLA_ROOT}:${AUTOVLA_ROOT}/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="${AUTOVLA_ROOT}/navsim"

AGENT_CONFIG_DIR="${AUTOVLA_ROOT}/navsim/navsim/planning/script/config/common/agent"
install -m 0644 "${DRIVEVLA_GUARD_ROOT}/configs/navsim_drivevla_guard_agent.yaml" \
  "${AGENT_CONFIG_DIR}/drivevla_guard_agent.yaml"

cd "${AUTOVLA_ROOT}"
python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_cot.py" \
  train_test_split=navtest \
  agent=drivevla_guard_agent \
  +agent.config_path="${AUTOVLA_MODEL_CONFIG}" \
  +agent.guard_config_path="${DRIVEVLA_GUARD_ROOT}/${GUARD_CONFIG}" \
  +agent.checkpoint_path="${AUTOVLA_CHECKPOINT}" \
  +agent.sensor_data_path="${NAVSIM_SENSOR_DATA}" \
  +agent.lora_conf.use_lora=false \
  metric_cache_path="${NAVSIM_METRIC_CACHE}" \
  json_data_path="${NAVSIM_JSON_DATA}" \
  experiment_name="drivevla_guard_${RUN_VARIANT}"
