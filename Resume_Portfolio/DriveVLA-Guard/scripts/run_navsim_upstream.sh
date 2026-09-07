#!/usr/bin/env bash
set -euo pipefail

: "${AUTOVLA_ROOT:?Set AUTOVLA_ROOT to the AutoVLA checkout}"
: "${DRIVEVLA_GUARD_ROOT:?Set DRIVEVLA_GUARD_ROOT to this project root}"
: "${AUTOVLA_CHECKPOINT:?Set AUTOVLA_CHECKPOINT to the released .ckpt}"
: "${QWEN_MODEL_ROOT:?Set QWEN_MODEL_ROOT to the local Qwen model directory}"
: "${AUTOVLA_MODEL_CONFIG:?Set AUTOVLA_MODEL_CONFIG to the rendered upstream YAML}"
: "${NAVSIM_JSON_DATA:?Set NAVSIM_JSON_DATA to preprocessed AutoVLA JSON data}"
: "${NAVSIM_SENSOR_DATA:?Set NAVSIM_SENSOR_DATA to NAVSIM sensor blobs}"
: "${NAVSIM_METRIC_CACHE:?Set NAVSIM_METRIC_CACHE to NAVSIM metric cache}"

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
  --output "${DRIVEVLA_GUARD_ROOT}/artifacts/official/preflight_upstream_b0.json"

export PYTHONPATH="${DRIVEVLA_GUARD_ROOT}/src:${AUTOVLA_ROOT}:${AUTOVLA_ROOT}/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="${AUTOVLA_ROOT}/navsim"
cd "${AUTOVLA_ROOT}"
python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_cot.py" \
  train_test_split=navtest \
  agent=autovla_agent \
  +agent.config_path="${AUTOVLA_MODEL_CONFIG}" \
  +agent.checkpoint_path="${AUTOVLA_CHECKPOINT}" \
  +agent.sensor_data_path="${NAVSIM_SENSOR_DATA}" \
  +agent.lora_conf.use_lora=false \
  metric_cache_path="${NAVSIM_METRIC_CACHE}" \
  json_data_path="${NAVSIM_JSON_DATA}" \
  experiment_name=autovla_upstream_b0
