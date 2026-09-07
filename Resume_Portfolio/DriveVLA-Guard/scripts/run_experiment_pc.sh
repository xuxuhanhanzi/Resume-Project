#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="${DRIVEVLA_GUARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_ROOT="${EXPERIMENT_ENVS_ROOT:-${HOME}/.venvs}"
ASSETS_ROOT="${EXPERIMENT_ASSETS_ROOT:-${HOME}/resume-project-assets}"
ENV_DIR="${DRIVEVLA_ENV:-${ENV_ROOT}/drivevla-guard}"
ASSET_DIR="${DRIVEVLA_ASSET_ROOT:-${ASSETS_ROOT}/drivevla-guard}"
PYTHON="${ENV_DIR}/bin/python"
RUN_ID="${EXPERIMENT_RUN_ID:-${PROFILE}-latest}"
SAFE_ID="${RUN_ID//[^A-Za-z0-9_.-]/_}"
STATE_DIR="${ROOT}/artifacts/official/${SAFE_ID}-steps"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
[[ -x "${PYTHON}" ]] || { echo "DriveVLA environment missing; run setup first" >&2; exit 2; }
mkdir -p "${STATE_DIR}"

run_step() {
  local name="$1"
  shift
  local marker="${STATE_DIR}/${name}.exit"
  if [[ -f "${marker}" && "$(cat "${marker}")" == "0" ]]; then
    echo "skip completed ${name}"
    return 0
  fi
  set +e
  "$@"
  local code=$?
  set -e
  echo "${code}" > "${marker}"
  return "${code}"
}

export DRIVEVLA_GUARD_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"
run_step quality "${PYTHON}" scripts/check.py

if [[ "${PROFILE}" == "smoke" ]]; then
  exit 0
fi

export AUTOVLA_ROOT="${AUTOVLA_ROOT:-${ASSET_DIR}/AutoVLA}"
export AUTOVLA_CHECKPOINT="${AUTOVLA_CHECKPOINT:-${ASSET_DIR}/models/AutoVLA/AutoVLA_PDMS_89.ckpt}"
export QWEN_MODEL_ROOT="${QWEN_MODEL_ROOT:-${ASSET_DIR}/models/Qwen2.5-VL-3B-Instruct}"
export NAVSIM_JSON_DATA="${NAVSIM_JSON_DATA:-${ASSET_DIR}/dataset/nuplan/navtest_nocot}"
export NAVSIM_SENSOR_DATA="${NAVSIM_SENSOR_DATA:-${ASSET_DIR}/dataset/nuplan/sensor_blobs/test}"
export NAVSIM_METRIC_CACHE="${NAVSIM_METRIC_CACHE:-${ASSET_DIR}/dataset/nuplan/navtest_metric_cache}"
export NUPLAN_MAPS_ROOT="${ASSET_DIR}/dataset/nuplan/maps"
export OPENSCENE_DATA_ROOT="${ASSET_DIR}/dataset/nuplan"
export NAVSIM_EXP_ROOT="${ASSET_DIR}/navsim-exp"

AUTOVLA_MODEL_CONFIG="${AUTOVLA_MODEL_CONFIG:-${ASSET_DIR}/state/autovla-eval.yaml}"
export AUTOVLA_MODEL_CONFIG
"${PYTHON}" scripts/prepare_autovla_eval_config.py \
  --template "${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-mix-sft.yaml" \
  --output "${AUTOVLA_MODEL_CONFIG}" \
  --qwen-model "${QWEN_MODEL_ROOT}" \
  --codebook "${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl" \
  --json-data "${NAVSIM_JSON_DATA}" \
  --sensor-data "${NAVSIM_SENSOR_DATA}" \
  --metric-cache "${NAVSIM_METRIC_CACHE}"

for repetition in 1 2; do
  echo "AutoVLA upstream B0 repetition ${repetition}"
  run_step "upstream_b0_rep${repetition}" bash scripts/run_navsim_upstream.sh
done
for variant in b0 e2; do
  export RUN_VARIANT="${variant}"
  run_step "guard_${variant}" bash scripts/run_navsim_guard.sh
done

run_step b0_e2_gate "${PYTHON}" scripts/check_navsim_gate.py \
  --upstream-root "${NAVSIM_EXP_ROOT}/autovla_upstream_b0" \
  --adapter-root "${NAVSIM_EXP_ROOT}/drivevla_guard_b0" \
  --e2-root "${NAVSIM_EXP_ROOT}/drivevla_guard_e2" \
  --output "${ROOT}/artifacts/official/${RUN_ID}_b0_e2_gate.json"

for variant in e3 e4; do
  export RUN_VARIANT="${variant}"
  run_step "guard_${variant}" bash scripts/run_navsim_guard.sh
done

run_step summary "${PYTHON}" scripts/summarize_navsim_results.py \
  --experiment-root "${NAVSIM_EXP_ROOT}" \
  --output "${ROOT}/artifacts/official/${RUN_ID}_summary.json"

echo "DriveVLA-Guard ${PROFILE} experiments complete for ${RUN_ID}."
