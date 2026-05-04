#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
DATASET_ROOT=${DATASET_ROOT:-"${PROJECT_ROOT}/data/splits_local"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/paper_ssl_benchmark"}
GPUS=${GPUS:-8}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

# Paper Table 2/3 downstream benchmark. PitVis is kept in splits as an auxiliary
# dataset, but it is not part of the 16-dataset paper matrix.
PAPER_DATASETS=(
  AlxSuture
  AutoLaparo
  cat-21
  cataract-101
  cataract-1k-phase
  cholec80
  Colonoscopic-web
  hyper-kvasir
  JIGSAWS
  kvasir-capsule
  LapGyn_dataset
  LDPolyVideo
  M2CAI16-Workflow
  MultiBypass140
  SAR-RARP50
  SurgicalActions160
)

PAPER_BACKBONES=(
  dinov3
  dinov3_surgenetxl
  videomae
  surgrec_mae
  jepa_vitl16
  surgrec_jepa
)

usage() {
  cat <<USAGE
Usage: $0 [--dry-run] [--datasets d1,d2] [--backbones b1,b2]

Environment:
  DATASET_ROOT, CKPT_ROOT, OUTPUT_ROOT, GPUS, CONTINUE_ON_ERROR

Backbones default to: ${PAPER_BACKBONES[*]}
Datasets default to the 16-dataset paper benchmark.
USAGE
}

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${csv}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "Missing --datasets value" >&2; exit 1; }
      split_csv "$2" PAPER_DATASETS
      shift 2
      ;;
    --backbones)
      [[ $# -ge 2 ]] || { echo "Missing --backbones value" >&2; exit 1; }
      split_csv "$2" PAPER_BACKBONES
      shift 2
      ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${DRY_RUN}" != 1 ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  SUMMARY="${OUTPUT_ROOT}/summary.tsv"
  printf 'dataset\tbackbone\tstatus\tlog\n' > "${SUMMARY}"
fi

for backbone in "${PAPER_BACKBONES[@]}"; do
  for dataset in "${PAPER_DATASETS[@]}"; do
    log="${OUTPUT_ROOT}/logs/${backbone}__${dataset}.log"
    cmd=(bash "${PROJECT_ROOT}/scripts/train/finetune_surgrec_video.sh" "${dataset}" --backbone "${backbone}")
    if [[ "${DRY_RUN}" == 1 ]]; then
      cmd+=(--dry-run)
    fi
    echo "[INFO] ${backbone} ${dataset}"
    if [[ "${DRY_RUN}" == 1 ]]; then
      DATASET_ROOT="${DATASET_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}/runs" GPUS="${GPUS}" "${cmd[@]}"
      continue
    fi
    if DATASET_ROOT="${DATASET_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}/runs" GPUS="${GPUS}" "${cmd[@]}" > "${log}" 2>&1; then
      printf '%s\t%s\tOK\t%s\n' "${dataset}" "${backbone}" "${log}" >> "${SUMMARY}"
    else
      code=$?
      printf '%s\t%s\tFAIL_%s\t%s\n' "${dataset}" "${backbone}" "${code}" "${log}" >> "${SUMMARY}"
      if [[ "${CONTINUE_ON_ERROR}" != 1 ]]; then
        exit "${code}"
      fi
    fi
  done
done
