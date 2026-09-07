#!/usr/bin/env bash
set -euo pipefail

ROOT="${FORGEMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_ENV="${FORGEMM_ENV:-${HOME}/.venvs/forgemm}"
RUNS="${FORGEMM_RUN_DIR:-${ROOT}/artifacts/runs/cloud_stage04}"
DATA_DIR="${RUNS}/data"

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${ROOT}/src"
cd "${ROOT}"
mkdir -p "${DATA_DIR}"

AUDIT="${DATA_DIR}/dataset_audit.json"
if [[ ! -f "${AUDIT}" ]]; then
  python scripts/audit_datasets.py --output "${AUDIT}"
fi

build_store() {
  local split="$1"
  local source="$2"
  local stem="${split}_${source}"
  local store="${DATA_DIR}/${stem}_evidence.jsonl"
  local summary="${DATA_DIR}/${stem}_evidence_summary.json"
  if [[ -f "${store}" && -f "${summary}" ]]; then
    echo "skip completed EvidenceStore ${stem}"
    return 0
  fi
  python scripts/build_evidence_store.py \
    --split "${split}" \
    --source "${source}" \
    --audit-report "${AUDIT}" \
    --output "${store}" \
    --summary "${summary}"
}

build_store train human
build_store train augmented
build_store val human
build_store test human

TRAIN_HUMAN="${DATA_DIR}/train_human_evidence.jsonl"
TRAIN_AUGMENTED="${DATA_DIR}/train_augmented_evidence.jsonl"
if [[ ! -f "${DATA_DIR}/structured_sft_train.jsonl" ]]; then
  python scripts/build_training_datasets.py \
    --evidence-store "${TRAIN_HUMAN}" \
    --evidence-store "${TRAIN_AUGMENTED}" \
    --sft-output "${DATA_DIR}/structured_sft_train.jsonl" \
    --grpo-output "${DATA_DIR}/grpo_train.jsonl" \
    --summary "${DATA_DIR}/structured_training_summary.json" \
    --portable-image-root "${ROOT}"
fi
if [[ ! -f "${DATA_DIR}/answer_sft_train.jsonl" ]]; then
  python scripts/build_answer_training_dataset.py \
    --evidence-store "${TRAIN_HUMAN}" \
    --evidence-store "${TRAIN_AUGMENTED}" \
    --output "${DATA_DIR}/answer_sft_train.jsonl" \
    --summary "${DATA_DIR}/answer_training_summary.json" \
    --portable-image-root "${ROOT}"
fi

build_strict_eval() {
  local split="$1"
  local store="${DATA_DIR}/${split}_human_evidence.jsonl"
  local output="${DATA_DIR}/chartqa_${split}_strict_eval.jsonl"
  if [[ ! -f "${output}" ]]; then
    python scripts/build_frozen_strict_eval.py \
      --evidence-store "${store}" \
      --output "${output}" \
      --summary "${DATA_DIR}/chartqa_${split}_strict_eval_summary.json" \
      --portable-image-root "${ROOT}"
  fi
}

build_strict_eval val
build_strict_eval test

if [[ ! -f "${DATA_DIR}/frozen_eval/frozen_eval_summary.json" ]]; then
  python scripts/build_frozen_eval_datasets.py \
    --output-dir "${DATA_DIR}/frozen_eval" \
    --project-root "${ROOT}"
fi
