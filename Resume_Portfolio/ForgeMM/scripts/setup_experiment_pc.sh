#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_ROOT="${EXPERIMENT_ENVS_ROOT:-${HOME}/.venvs}"
ASSETS_ROOT="${EXPERIMENT_ASSETS_ROOT:-${HOME}/resume-project-assets}"
export FORGEMM_ENV="${FORGEMM_ENV:-${ENV_ROOT}/forgemm}"
MODEL_ROOT="${FORGEMM_MODEL:-${ASSETS_ROOT}/forgemm/models/Qwen2.5-VL-3B-Instruct}"
QWEN_REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
mkdir -p "${ENV_ROOT}" "${ASSETS_ROOT}/forgemm/models"
FORGEMM_ROOT="${ROOT}" bash "${ROOT}/scripts/setup_local_wsl.sh"
"${FORGEMM_ENV}/bin/uv" pip install --python "${FORGEMM_ENV}/bin/python" huggingface_hub 2>/dev/null || \
  uv pip install --python "${FORGEMM_ENV}/bin/python" huggingface_hub
"${FORGEMM_ENV}/bin/hf" download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${QWEN_REVISION}" --local-dir "${MODEL_ROOT}"

CONTROLLED_DATASET="${ROOT}/datasets/ForgeMM-Controlled-v1-Lite"
if [[ ! -f "${CONTROLLED_DATASET}/manifests/train_oracle.jsonl" ]]; then
  "${FORGEMM_ENV}/bin/python" "${ROOT}/scripts/build_controlled_dataset.py" \
    --output "${CONTROLLED_DATASET}" --project-root "${ROOT}" --seed 20260824
fi
"${FORGEMM_ENV}/bin/python" "${ROOT}/scripts/audit_controlled_dataset.py" \
  --dataset "${CONTROLLED_DATASET}" \
  --output "${ROOT}/artifacts/runs/experiment_pc_setup/controlled_dataset_audit.json"

if [[ "${PROFILE}" == "full" ]]; then
  "${FORGEMM_ENV}/bin/python" "${ROOT}/scripts/download_datasets.py" \
    --project-root "${ROOT}" --cache-root "${ASSETS_ROOT}/forgemm"
fi
"${FORGEMM_ENV}/bin/python" "${ROOT}/scripts/local_lite_runtime_audit.py" \
  --output "${ROOT}/artifacts/runs/experiment_pc_setup/runtime_audit.json"
