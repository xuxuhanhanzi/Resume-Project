#!/usr/bin/env bash
set -euo pipefail

# Run inside an Ubuntu WSL2 shell after WSL itself has been installed.
ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USER_HOME_DIR="$(getent passwd "$(id -u)" | awk -F: '{print $6}')"
UV_BIN="${FORGEMM_UV:-${USER_HOME_DIR}/.local/bin/uv}"
ENV_DIR="${FORGEMM_ENV:-${USER_HOME_DIR}/.venvs/forgemm}"
PYTHON_VERSION="${FORGEMM_PYTHON_VERSION:-3.12}"

python3 --version
nvidia-smi

if [[ ! -x "${UV_BIN}" ]]; then
  printf 'Missing uv at %s. Install it in this WSL user account before continuing.\n' "${UV_BIN}" >&2
  exit 2
fi

# Ubuntu 26.04 currently supplies Python 3.14, while this project explicitly
# supports Python 3.10--3.12.  uv installs the compatible interpreter in the
# user account; keeping the virtual environment on the Linux filesystem also
# avoids placing Linux symlinks and wheels on the mounted Windows drive.
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${ENV_DIR}"
fi

# CUDA wheel selection remains visible in the saved runtime audit.  Do not use
# a global Windows interpreter or mix it with this Linux environment.
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  -r "${ROOT}/requirements-dev.lock" -r "${ROOT}/requirements-swift.in"
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  "transformers>=4.49" "qwen-vl-utils>=0.0.6"

"${ENV_DIR}/bin/python" "${ROOT}/scripts/local_lite_runtime_audit.py" \
  --output "${ROOT}/artifacts/runs/local_controlled_v1_lite/runtime_audit.json"
