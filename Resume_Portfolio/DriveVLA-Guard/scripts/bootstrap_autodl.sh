#!/usr/bin/env bash
set -euo pipefail

: "${DRIVEVLA_GUARD_ROOT:?Set DRIVEVLA_GUARD_ROOT}"
: "${AUTOVLA_ROOT:?Set AUTOVLA_ROOT}"

python -m pip install -e "${DRIVEVLA_GUARD_ROOT}"
python -m pip install -e "${AUTOVLA_ROOT}" --no-warn-conflicts
python -m pip install -e "${AUTOVLA_ROOT}/navsim" --no-warn-conflicts

python - <<'PY'
import sys
import torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

