#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${EXPERIMENT_RUN_ID:-${PROFILE}-latest}"
RUN_DIR="${ROOT}/docs/experiments/artifacts/experiment-pc/${RUN_ID}"
case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
mkdir -p "${RUN_DIR}"
export HOSPITAL_API_URL="http://localhost:${BACKEND_PORT:-8080}"
export HOSPITAL_KEYCLOAK_URL="http://localhost:${KEYCLOAK_PORT:-8081}"
export HOSPITAL_PROMETHEUS_URL="http://localhost:${PROMETHEUS_PORT:-9090}"
export HOSPITAL_GRAFANA_URL="http://localhost:${GRAFANA_PORT:-3000}"
export HOSPITAL_FRONTEND_URL="http://localhost:${FRONTEND_PORT:-5173}"
export K6_RESULTS_DIR="${RUN_DIR}"
cd "${ROOT}"

finish() {
  local code=$?
  trap - EXIT
  docker compose logs --no-color > "${RUN_DIR}/compose.log" 2>&1 || true
  docker compose down || true
  exit "${code}"
}
trap finish EXIT

docker compose up -d
python3 scripts/demo.py --output "${RUN_DIR}/acceptance.json"
if [[ "${PROFILE}" == "smoke" ]]; then
  export K6_SMOKE=true
else
  export K6_SMOKE=false
fi
docker compose --profile perf run --rm load-test
docker compose ps --format json > "${RUN_DIR}/compose-ps.json"
trap - EXIT
docker compose logs --no-color > "${RUN_DIR}/compose.log" 2>&1
docker compose down
