#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="${DRIVEVLA_GUARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_ROOT="${EXPERIMENT_ENVS_ROOT:-${HOME}/.venvs}"
ASSETS_ROOT="${EXPERIMENT_ASSETS_ROOT:-${HOME}/resume-project-assets}"
ENV_DIR="${DRIVEVLA_ENV:-${ENV_ROOT}/drivevla-guard}"
ASSET_DIR="${DRIVEVLA_ASSET_ROOT:-${ASSETS_ROOT}/drivevla-guard}"
AUTOVLA_ROOT="${AUTOVLA_ROOT:-${ASSET_DIR}/AutoVLA}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
AUTOVLA_COMMIT="ba34eed74ce6729e7986592d0e66cbaca397b4fa"
QWEN_REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"
CHECKPOINT_REVISION="a7d7ba3ed7529b248d2694c2defa31b35208340f"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
[[ -n "${UV_BIN}" ]] || { echo "uv is missing; run the portfolio bootstrap first" >&2; exit 2; }

mkdir -p "${ENV_ROOT}" "${ASSET_DIR}" "${ASSET_DIR}/state"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.10 "${ENV_DIR}"
fi
PYTHON="${ENV_DIR}/bin/python"
"${UV_BIN}" pip install --python "${PYTHON}" -r "${ROOT}/requirements-dev.lock" -e "${ROOT}"

if [[ "${PROFILE}" == "smoke" ]]; then
  "${PYTHON}" -c 'import drivevla_guard; print("DriveVLA-Guard smoke environment ready")'
  exit 0
fi

if [[ "${DRIVEVLA_ACCEPT_OPENSCENE_LICENSE:-0}" != "1" ]]; then
  echo "Full NAVSIM setup is gated by the OpenScene CC BY-NC-SA 4.0 terms." >&2
  echo "Review https://huggingface.co/datasets/OpenDriveLab/OpenScene and rerun with DRIVEVLA_ACCEPT_OPENSCENE_LICENSE=1." >&2
  exit 3
fi

if [[ ! -d "${AUTOVLA_ROOT}/.git" ]]; then
  git clone https://github.com/ucla-mobility/AutoVLA.git "${AUTOVLA_ROOT}"
fi
git -C "${AUTOVLA_ROOT}" fetch --depth 1 origin "${AUTOVLA_COMMIT}"
git -C "${AUTOVLA_ROOT}" checkout --detach "${AUTOVLA_COMMIT}"

# The upstream lock pins CUDA 12.4-era torch/triton builds.  Filter only those
# three exact pins, then install the Blackwell-capable CUDA 12.8 wheel explicitly.
awk '!/^torch==/ && !/^torchvision==/ && !/^triton==/' \
  "${AUTOVLA_ROOT}/requirements.txt" > "${ASSET_DIR}/state/autovla-requirements-cu128.txt"
"${UV_BIN}" pip install --python "${PYTHON}" \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"${UV_BIN}" pip install --python "${PYTHON}" \
  -r "${ASSET_DIR}/state/autovla-requirements-cu128.txt" huggingface_hub
"${UV_BIN}" pip install --python "${PYTHON}" -e "${AUTOVLA_ROOT}" -e "${AUTOVLA_ROOT}/navsim" -e "${ROOT}"

QWEN_MODEL_ROOT="${ASSET_DIR}/models/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_DIR="${ASSET_DIR}/models/AutoVLA"
mkdir -p "${QWEN_MODEL_ROOT}" "${CHECKPOINT_DIR}"
"${ENV_DIR}/bin/hf" download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${QWEN_REVISION}" --local-dir "${QWEN_MODEL_ROOT}"
"${ENV_DIR}/bin/hf" download Zewei-Zhou/AutoVLA AutoVLA_PDMS_89.ckpt \
  --revision "${CHECKPOINT_REVISION}" --local-dir "${CHECKPOINT_DIR}"
echo "58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473  ${CHECKPOINT_DIR}/AutoVLA_PDMS_89.ckpt" \
  | sha256sum --check --strict
"${PYTHON}" "${ROOT}/scripts/download_navsim_assets.py" --asset-root "${ASSET_DIR}"

DATA_ROOT="${ASSET_DIR}/dataset/nuplan"
CONFIG_PATH="${AUTOVLA_ROOT}/config/dataset/drivevla_experiment_pc.yaml"
"${PYTHON}" - "${CONFIG_PATH}" "${QWEN_MODEL_ROOT}" "${DATA_ROOT}" "${AUTOVLA_ROOT}" <<'PY'
import sys, yaml
from pathlib import Path
path, model, data, upstream = map(Path, sys.argv[1:])
payload = {
    "name": "drivevla-experiment-pc",
    "description": "Pinned navtest no-CoT preprocessing for DriveVLA-Guard",
    "pretrained_model_path": str(model),
    "batch_size": 1,
    "num_workers": 12,
    "dataset_name": "nuplan",
    "dataset_path": str(data / "test_navsim_logs"),
    "scene_filter": str(upstream / "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"),
}
path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

export PYTHONPATH="${AUTOVLA_ROOT}:${AUTOVLA_ROOT}/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="${AUTOVLA_ROOT}/navsim"
export OPENSCENE_DATA_ROOT="${DATA_ROOT}"
export NUPLAN_MAPS_ROOT="${DATA_ROOT}/maps"
export NAVSIM_EXP_ROOT="${ASSET_DIR}/navsim-exp"

if [[ ! -d "${DATA_ROOT}/navtest_metric_cache" ]]; then
  "${PYTHON}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest cache.cache_path="${DATA_ROOT}/navtest_metric_cache"
fi
if [[ ! -d "${DATA_ROOT}/navtest_nocot" ]]; then
  cd "${AUTOVLA_ROOT}"
  "${PYTHON}" tools/preprocessing/nocot_sample_generation.py \
    --config dataset/drivevla_experiment_pc --output_dir "${DATA_ROOT}/navtest_nocot" --num_workers 12
fi
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
