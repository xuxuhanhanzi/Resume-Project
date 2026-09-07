#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-smoke}"
RUN_ID="${EXPERIMENT_RUN_ID:-${PROFILE}-latest}"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac

if [[ ! -r /proc/version ]] || ! grep -qi microsoft /proc/version; then
  echo "This entry point must run inside Ubuntu on WSL2." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv for the current WSL user..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${ROOT}"
python3 scripts/experiment_suite.py setup --profile "${PROFILE}" --run-id "${RUN_ID}"
