#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-smoke}"
RUN_ID="${EXPERIMENT_RUN_ID:-${PROFILE}-latest}"

case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac

cd "${ROOT}"
python3 scripts/experiment_suite.py run --profile "${PROFILE}" --run-id "${RUN_ID}"
