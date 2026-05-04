#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [--dry-run] <dataset-name> <qwen3|qwen25|qwen2.5|llava-next> [extra args...]" >&2
  exit 1
fi

DATASET="$1"
MODEL_KIND="$2"
shift 2

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)

abspath_from_project() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ "${path}" = /* ]]; then
    printf '%s
' "${path}"
  else
    printf '%s
' "${PROJECT_ROOT}/${path}"
  fi
}

DATASET_ROOT=${DATASET_ROOT:-"${PROJECT_ROOT}/data/splits_local"}
MODEL_ROOT=${MODEL_ROOT:-"${PROJECT_ROOT}/checkpoints"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/vlm_eval"}
DATASET_ROOT=$(abspath_from_project "${DATASET_ROOT}")
MODEL_ROOT=$(abspath_from_project "${MODEL_ROOT}")
OUTPUT_ROOT=$(abspath_from_project "${OUTPUT_ROOT}")
SINGLE_DATASET_ROOT="${OUTPUT_ROOT}/_dataset_roots/${DATASET}"
SCAN_ROOT=$(dirname "${SINGLE_DATASET_ROOT}")

if [[ ! -d "${DATASET_ROOT}/${DATASET}" ]]; then
  echo "Missing dataset split directory: ${DATASET_ROOT}/${DATASET}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}" "${SCAN_ROOT}"
rm -f "${SINGLE_DATASET_ROOT}"
ln -s "${DATASET_ROOT}/${DATASET}" "${SINGLE_DATASET_ROOT}"

case "${MODEL_KIND}" in
  qwen3)
    CMD=(python "${PROJECT_ROOT}/vlm_eval/run_qwen3_vlm_test.py"
      --dataset-root "${SCAN_ROOT}"
      --model-path "${MODEL_ROOT}/Qwen3-VL-8B-Instruct"
      --output-root "${OUTPUT_ROOT}/qwen3"
      "$@")
    ;;
  qwen25|qwen2.5|qwen2_5)
    CMD=(python "${PROJECT_ROOT}/vlm_eval/run_qwen3_vlm_test.py"
      --dataset-root "${SCAN_ROOT}"
      --model-path "${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct"
      --output-root "${OUTPUT_ROOT}/qwen25"
      "$@")
    ;;
  llava-next)
    CMD=(python "${PROJECT_ROOT}/vlm_eval/run_llava_next_test.py"
      --dataset-root "${SCAN_ROOT}"
      --model-path "${MODEL_ROOT}/LLaVA-NeXT-vicuna-7b"
      --output-root "${OUTPUT_ROOT}/llava-next"
      "$@")
    ;;
  *)
    echo "Unknown VLM kind: ${MODEL_KIND}" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
