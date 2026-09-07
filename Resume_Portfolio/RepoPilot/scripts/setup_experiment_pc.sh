#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${EXPERIMENT_ENVS_ROOT:-${HOME}/.venvs}"
ENV_DIR="${REPOPILOT_ENV:-${ENV_ROOT}/repopilot}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
STATE_ROOT="${EXPERIMENT_ASSETS_ROOT:-${HOME}/resume-project-assets}/repopilot"
REVISION="a637bd46829f3132e12938c8a0ca93173a977b8e"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
[[ -n "${UV_BIN}" ]] || { echo "uv is missing; run the portfolio bootstrap first" >&2; exit 2; }
mkdir -p "${ENV_ROOT}" "${STATE_ROOT}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.12 "${ENV_DIR}"
fi
PYTHON="${ENV_DIR}/bin/python"
"${UV_BIN}" pip install --python "${PYTHON}" -r "${ROOT}/requirements-dev.lock" -e "${ROOT}[service]" \
  huggingface_hub pandas==2.2.3 pyarrow==18.1.0

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  nohup ollama serve > "${STATE_ROOT}/ollama.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -fsS http://127.0.0.1:11434/api/version >/dev/null
ollama pull qwen2.5:7b

"${PYTHON}" "${ROOT}/scripts/download_benchmark_data.py" --project-root "${ROOT}" --profile "${PROFILE}"
COUNT=3
[[ "${PROFILE}" == "full" ]] && COUNT=60
"${PYTHON}" "${ROOT}/scripts/materialize_frames_corpus.py" --count "${COUNT}"

docker build -t repopilot-dabench:py311-v1 -f "${ROOT}/deploy/docker/sandbox.Dockerfile" "${ROOT}"
if [[ "${PROFILE}" == "full" ]]; then
  DATA_ROOT="${ROOT}/evaluation/benchmarks/data/swebench_live/${REVISION}"
  WORKSPACE_ROOT="${ROOT}/evaluation/benchmarks/workspaces/swebench_live/${REVISION}"
  OUT_DIR="${DATA_ROOT}/fresh5"
  "${PYTHON}" "${ROOT}/scripts/prepare_swebench_fresh_holdout.py" \
    --parquet "${DATA_ROOT}/lite.parquet" --count 5 --seed 7 \
    --workspace-root "${WORKSPACE_ROOT}" --out-dir "${OUT_DIR}" --clone
  "${PYTHON}" - "${OUT_DIR}/fresh_task_config.json" <<'PY' | while IFS= read -r image; do docker pull "${image}"; done
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8")).values():
    print(item["image"])
PY
fi
