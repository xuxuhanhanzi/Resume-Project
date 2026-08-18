#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTOVLA_DATA_ROOT:-/root/autodl-tmp}"
MODEL_FILE="${DATA_ROOT}/models/AutoVLA_PDMS_89.ckpt.download"
MODEL_SHA256="58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473"
MODEL_URL="https://hf-mirror.com/Zewei-Zhou/AutoVLA/resolve/main/AutoVLA_PDMS_89.ckpt?download=true"
WARMUP_FILE="${DATA_ROOT}/navsim_v2.2_warmup_two_stage.tar.gz.download"
WARMUP_SHA256="b649385aacffe65824d9ab609f45c1fcd0ad258b4c03e7cd52d26d096425c904"
WARMUP_URL="https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz?download=true"

while [[ ! -f "${DATA_ROOT}/AutoVLA-checkpoint-loop.exit" ]]; do
  sleep 30
done

if [[ "$(cat "${DATA_ROOT}/AutoVLA-checkpoint-loop.exit")" != "0" ]]; then
  code=1
  for attempt in $(seq 1 100); do
    echo "RESCUE_ATTEMPT ${attempt} $(date -Is)" >> "${DATA_ROOT}/AutoVLA-checkpoint-rescue.log"
    set +e
    timeout -s INT 300 aria2c -c -x16 -s32 -k1M \
      --file-allocation=none --max-tries=0 --retry-wait=3 --timeout=90 \
      --connect-timeout=20 --auto-file-renaming=false --allow-overwrite=true \
      -d "${DATA_ROOT}/models" -o "$(basename "${MODEL_FILE}")" "${MODEL_URL}" \
      >> "${DATA_ROOT}/AutoVLA-checkpoint-rescue.log" 2>&1
    code=$?
    set -e
    [[ "${code}" == "0" ]] && break
    sleep 3
  done
  echo "${code}" > "${DATA_ROOT}/AutoVLA-checkpoint-loop.exit"
  [[ "${code}" == "0" ]] || exit "${code}"

  sha256sum "${MODEL_FILE}" > "${DATA_ROOT}/AutoVLA-checkpoint-loop.sha256"
  actual="$(cut -d' ' -f1 "${DATA_ROOT}/AutoVLA-checkpoint-loop.sha256")"
  if [[ "${actual}" != "${MODEL_SHA256}" ]]; then
    echo 3 > "${DATA_ROOT}/AutoVLA-checkpoint-loop-verify.exit"
    exit 3
  fi
  echo 0 > "${DATA_ROOT}/AutoVLA-checkpoint-loop-verify.exit"

  source "${DATA_ROOT}/envs/autovla310/bin/activate"
  cd "${DATA_ROOT}/AutoVLA"
  set +e
  python -m pip install -r requirements_no_nuplan.txt \
    > "${DATA_ROOT}/AutoVLA-deps-no-nuplan-rescue.log" 2>&1
  code=$?
  set -e
  echo "${code}" > "${DATA_ROOT}/AutoVLA-deps-no-nuplan-loop.exit"
  [[ "${code}" == "0" ]] || exit "${code}"

  code=1
  for attempt in $(seq 1 30); do
    echo "RESCUE_ATTEMPT ${attempt} $(date -Is)" >> "${DATA_ROOT}/navsim-warmup-download.log"
    set +e
    timeout -s INT 300 aria2c -c -x16 -s32 -k1M \
      --file-allocation=none --max-tries=0 --retry-wait=3 --timeout=90 \
      --connect-timeout=20 --auto-file-renaming=false --allow-overwrite=true \
      -d "${DATA_ROOT}" -o "$(basename "${WARMUP_FILE}")" "${WARMUP_URL}" \
      >> "${DATA_ROOT}/navsim-warmup-download.log" 2>&1
    code=$?
    set -e
    [[ "${code}" == "0" ]] && break
    sleep 3
  done
  echo "${code}" > "${DATA_ROOT}/navsim-warmup-download.exit"
  [[ "${code}" == "0" ]] || exit "${code}"

  sha256sum "${WARMUP_FILE}" > "${DATA_ROOT}/navsim-warmup-download.sha256"
  actual="$(cut -d' ' -f1 "${DATA_ROOT}/navsim-warmup-download.sha256")"
  if [[ "${actual}" != "${WARMUP_SHA256}" ]]; then
    echo 3 > "${DATA_ROOT}/navsim-warmup-verify.exit"
    exit 3
  fi
  echo 0 > "${DATA_ROOT}/navsim-warmup-verify.exit"
  tar -tzf "${WARMUP_FILE}" > "${DATA_ROOT}/navsim-warmup-members.txt"
  echo 0 > "${DATA_ROOT}/navsim-warmup-tar-test.exit"
fi
