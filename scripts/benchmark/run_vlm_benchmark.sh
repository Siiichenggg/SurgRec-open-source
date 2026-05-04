#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
DATASET_ROOT=${DATASET_ROOT:-"${PROJECT_ROOT}/data/splits_local"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/paper_vlm_benchmark"}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

PAPER_DATASETS=(
  AlxSuture AutoLaparo cat-21 cataract-101 cataract-1k-phase cholec80
  Colonoscopic-web hyper-kvasir JIGSAWS kvasir-capsule LapGyn_dataset
  LDPolyVideo M2CAI16-Workflow MultiBypass140 SAR-RARP50 SurgicalActions160
)
PAPER_VLMS=(qwen3 llava-next qwen25)
PROMPTS=(baseline stability_v1)

usage() {
  echo "Usage: $0 [--dry-run] [--prompt baseline|stability_v1] [--vlms qwen3,llava-next,qwen25]"
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
    --prompt)
      [[ $# -ge 2 ]] || { echo "Missing --prompt value" >&2; exit 1; }
      PROMPTS=("$2")
      shift 2
      ;;
    --vlms)
      [[ $# -ge 2 ]] || { echo "Missing --vlms value" >&2; exit 1; }
      split_csv "$2" PAPER_VLMS
      shift 2
      ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${DRY_RUN}" != 1 ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  SUMMARY="${OUTPUT_ROOT}/summary.tsv"
  printf 'dataset\tvlm\tprompt\tstatus\tlog\n' > "${SUMMARY}"
fi

for vlm in "${PAPER_VLMS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    for dataset in "${PAPER_DATASETS[@]}"; do
      log="${OUTPUT_ROOT}/logs/${vlm}__${prompt}__${dataset}.log"
      cmd=(bash "${PROJECT_ROOT}/scripts/eval/evaluate_vlm.sh" "${dataset}" "${vlm}" --prompt-variant "${prompt}")
      if [[ "${DRY_RUN}" == 1 ]]; then
        cmd=(bash "${PROJECT_ROOT}/scripts/eval/evaluate_vlm.sh" --dry-run "${dataset}" "${vlm}" --prompt-variant "${prompt}")
      fi
      echo "[INFO] ${vlm} ${prompt} ${dataset}"
      if [[ "${DRY_RUN}" == 1 ]]; then
        DATASET_ROOT="${DATASET_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}/runs" "${cmd[@]}"
        continue
      fi
      if DATASET_ROOT="${DATASET_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}/runs" "${cmd[@]}" > "${log}" 2>&1; then
        printf '%s\t%s\t%s\tOK\t%s\n' "${dataset}" "${vlm}" "${prompt}" "${log}" >> "${SUMMARY}"
      else
        code=$?
        printf '%s\t%s\t%s\tFAIL_%s\t%s\n' "${dataset}" "${vlm}" "${prompt}" "${code}" "${log}" >> "${SUMMARY}"
        if [[ "${CONTINUE_ON_ERROR}" != 1 ]]; then
          exit "${code}"
        fi
      fi
    done
  done
done
