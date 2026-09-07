#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_ROOT="${EXPERIMENT_ENVS_ROOT:-${HOME}/.venvs}"
ASSETS_ROOT="${EXPERIMENT_ASSETS_ROOT:-${HOME}/resume-project-assets}"
RUN_ID="${EXPERIMENT_RUN_ID:-${PROFILE}-latest}"
export FORGEMM_ROOT="${ROOT}"
export FORGEMM_ENV="${FORGEMM_ENV:-${ENV_ROOT}/forgemm}"
export FORGEMM_MODEL="${FORGEMM_MODEL:-${ASSETS_ROOT}/forgemm/models/Qwen2.5-VL-3B-Instruct}"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
[[ -x "${FORGEMM_ENV}/bin/python" ]] || { echo "ForgeMM environment missing; run setup first" >&2; exit 2; }

cd "${ROOT}"
source "${FORGEMM_ENV}/bin/activate"
python scripts/dev.py check
if [[ "${PROFILE}" == "smoke" ]]; then
  export FORGEMM_RUN_DIR="${ROOT}/artifacts/runs/experiment_pc/${RUN_ID}/smoke"
  bash scripts/run_local_lite_smoke.sh
else
  export FORGEMM_RUN_DIR="${ROOT}/artifacts/runs/experiment_pc/${RUN_ID}/stage04"
  bash scripts/run_cloud_stage04.sh
fi
