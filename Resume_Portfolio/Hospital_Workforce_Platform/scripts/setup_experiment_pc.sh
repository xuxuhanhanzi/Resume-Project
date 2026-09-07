#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${PROFILE}" in smoke|full) ;; *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;; esac
command -v docker >/dev/null 2>&1 || { echo "Docker is missing" >&2; exit 2; }
docker info >/dev/null
cd "${ROOT}"
docker compose config --quiet
docker compose pull --ignore-buildable
docker compose build
