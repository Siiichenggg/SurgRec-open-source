#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
TRAINING_DIR=${TRAINING_DIR:-"${PROJECT_ROOT}"}
DATASET_ROOT=${DATASET_ROOT:-"${PROJECT_ROOT}/data/splits_local"}
CKPT_ROOT=${CKPT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/finetune"}
GPUS=${GPUS:-8}
NUM_FRAMES=${NUM_FRAMES:-16}
SAMPLING_RATE=${SAMPLING_RATE:-4}
BACKBONE=surgrec_mae
DRY_RUN=0
LIST_ONLY=0
LIST_BACKBONES=0
DATASET=""
PRETRAIN_CKPT_VALUE=${PRETRAIN_CKPT:-}
BATCH_SIZE_VALUE=${BATCH_SIZE:-}
EPOCHS_VALUE=${EPOCHS:-}
LR_VALUE=${LR:-}
MODEL_NAME_VALUE=${MODEL_NAME:-}
EXTRA_ARGS=()
BACKBONE_ARGS=()
ENTRYPOINT_MODULE=""
ENTRYPOINT_PATH=""
MODEL_NAME=""
DEFAULT_CKPT=""
DEFAULT_BATCH_SIZE=""
DEFAULT_EPOCHS="50"
DEFAULT_LR=""

abspath_from_project() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

TRAINING_DIR=$(abspath_from_project "${TRAINING_DIR}")
DATASET_ROOT=$(abspath_from_project "${DATASET_ROOT}")
CKPT_ROOT=$(abspath_from_project "${CKPT_ROOT}")
OUTPUT_ROOT=$(abspath_from_project "${OUTPUT_ROOT}")

usage() {
  cat <<USAGE
Usage: $0 <dataset-name> [--backbone NAME] [--dry-run] [--list] [--list-backbones] [-- extra training args]

Paper backbones:
  surgrec_mae        SurgRec-MAE / balanced VideoMAE-style checkpoint
  surgrec_jepa       SurgRec-JEPA checkpoint fine-tuned with the VideoMAE ViT-L head
  videomae           General VideoMAE baseline
  videomaev2         VideoMAE V2-style baseline
  jepa_vitl16        Generic JEPA ViT-L baseline
  dinov3             DINOv3 ViT-L/16 baseline checkpoint
  dinov3_surgenetxl  DINOv3 ViT-L/16 SurgeNetXL checkpoint

Extra baselines kept for comparison:
  dino               Legacy DINO ResNet50 video wrapper
  endofm             EndoFM fine-tuning entrypoint
  mocov2             MoCo v2 frozen-backbone entrypoint

Environment overrides:
  DATASET_ROOT, CKPT_ROOT, PRETRAIN_CKPT, OUTPUT_ROOT, GPUS,
  BATCH_SIZE, EPOCHS, LR, NUM_FRAMES, SAMPLING_RATE, MODEL_NAME

Examples:
  $0 cholec80 --backbone surgrec_mae
  $0 AutoLaparo --backbone surgrec_jepa
  $0 cholec80 --backbone dinov3_surgenetxl -- --nb_classes 7
USAGE
}

list_datasets() {
  if [[ -d "${DATASET_ROOT}" ]]; then
    find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  else
    echo "Dataset root not found: ${DATASET_ROOT}" >&2
    return 1
  fi
}

list_backbones() {
  printf '%s\n' surgrec_mae surgrec_jepa videomae videomaev2 jepa_vitl16 dinov3 dinov3_surgenetxl dino endofm mocov2
}

normalize_dataset_key() {
  local input="$1"
  local lower="${input,,}"
  case "${lower}" in
    hyper-kvasir|hyper_kvasir) echo "Hyper-kvasir" ;;
    colonoscopic-web|colonoscopic_web) echo "colonoscopic_web" ;;
    cat-21|cat21) echo "OOD_cat-21" ;;
    cataract-101|cataract101) echo "OOD_cataract-101" ;;
    cataract-1k-phase|cataract1k|cataract-1k) echo "OOD_cataract-1k" ;;
    lapgyn|lapgyn_dataset|lapgyn-dataset) echo "LapGyn_dataset" ;;
    *) echo "${input}" ;;
  esac
}

configure_vit_video_backbone() {
  BACKBONE="$1"
  MODEL_NAME="${MODEL_NAME_VALUE:-$2}"
  DEFAULT_CKPT="${CKPT_ROOT}/$3"
  DEFAULT_BATCH_SIZE="$4"
  DEFAULT_LR="$5"
  ENTRYPOINT_MODULE="surgrec_video.entrypoints.run_class_finetuning_videomae"
  ENTRYPOINT_PATH="${PROJECT_ROOT}/surgrec_video/entrypoints/run_class_finetuning_videomae.py"
  BACKBONE_ARGS=(--backbone_variant videomaev2)
}

configure_backbone() {
  local key="${1,,}"
  BACKBONE_ARGS=()
  case "${key}" in
    surgrec_mae|sr-mae|sr_mae)
      configure_vit_video_backbone "surgrec_mae" "vit_base_patch16_224" "surgrec_mae.pth" "2" "5e-4"
      ;;
    surgrec_jepa|sr-jepa|sr_jepa|jepa_60w|jepa-v3-60w|jepa_v3_60w)
      configure_vit_video_backbone "surgrec_jepa" "vit_large_patch16_224" "surgrec_jepa.pt" "2" "5e-4"
      ;;
    videomaev2)
      configure_vit_video_backbone "videomaev2" "vit_base_patch16_224" "videomaev2.pth" "2" "5e-4"
      ;;
    videomae|videomae_e149)
      BACKBONE="videomae"
      ENTRYPOINT_MODULE="surgrec_video.entrypoints.run_class_finetuning_videomae"
      ENTRYPOINT_PATH="${PROJECT_ROOT}/surgrec_video/entrypoints/run_class_finetuning_videomae.py"
      MODEL_NAME="${MODEL_NAME_VALUE:-vit_base_patch16_224}"
      DEFAULT_CKPT="${CKPT_ROOT}/videomae.pth"
      DEFAULT_BATCH_SIZE="2"
      DEFAULT_LR="5e-4"
      BACKBONE_ARGS=(--backbone_variant videomae)
      ;;
    jepa_vitl16|jepa|vjepa)
      configure_vit_video_backbone "jepa_vitl16" "vit_large_patch16_224" "jepa_vitl16.pth.tar" "2" "5e-4"
      ;;
    dinov3|dinov3_vitl16)
      configure_vit_video_backbone "dinov3" "vit_large_patch16_224" "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" "2" "5e-4"
      ;;
    dinov3_surgenetxl|dinov3-surg|dinov3_surg)
      configure_vit_video_backbone "dinov3_surgenetxl" "vit_large_patch16_224" "DINOv3_ViTl16_size336_SurgeNetXL.pth" "2" "5e-4"
      ;;
    dino)
      BACKBONE="dino"
      ENTRYPOINT_MODULE="surgrec_video.entrypoints.run_class_finetuning_medical"
      ENTRYPOINT_PATH="${PROJECT_ROOT}/surgrec_video/entrypoints/run_class_finetuning_medical.py"
      MODEL_NAME="${MODEL_NAME_VALUE:-dino_resnet50_patch16_224}"
      DEFAULT_CKPT="${CKPT_ROOT}/dino.pth"
      DEFAULT_BATCH_SIZE="12"
      DEFAULT_LR="2e-5"
      BACKBONE_ARGS=(--backbone_variant dino --frozen_backbone)
      ;;
    endofm)
      BACKBONE="endofm"
      ENTRYPOINT_MODULE="surgrec_video.entrypoints.run_class_finetuning_medical"
      ENTRYPOINT_PATH="${PROJECT_ROOT}/surgrec_video/entrypoints/run_class_finetuning_medical.py"
      MODEL_NAME="${MODEL_NAME_VALUE:-endofm_vit_base_patch16_224}"
      DEFAULT_CKPT="${CKPT_ROOT}/endofm.pth"
      DEFAULT_BATCH_SIZE="12"
      DEFAULT_LR="2e-5"
      BACKBONE_ARGS=(--backbone_variant endofm --layer_decay 0.85 --drop_path 0.1 --weight_decay 0.05 --warmup_epochs 5 --tubelet_size 2 --use_checkpoint --use_mean_pooling)
      ;;
    mocov2)
      BACKBONE="mocov2"
      ENTRYPOINT_MODULE="surgrec_video.entrypoints.run_class_finetuning_medical"
      ENTRYPOINT_PATH="${PROJECT_ROOT}/surgrec_video/entrypoints/run_class_finetuning_medical.py"
      MODEL_NAME="${MODEL_NAME_VALUE:-mocov2_resnet50_patch16_224}"
      DEFAULT_CKPT="${CKPT_ROOT}/mocov2.pth"
      DEFAULT_BATCH_SIZE="12"
      DEFAULT_LR="2e-5"
      BACKBONE_ARGS=(--backbone_variant mocov2 --frozen_backbone)
      ;;
    *)
      echo "Unknown backbone: ${1}" >&2
      echo "Supported backbones: $(list_backbones | tr '\n' ' ')" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --list) LIST_ONLY=1; shift ;;
    --list-backbones) LIST_BACKBONES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --backbone)
      [[ $# -ge 2 ]] || { echo "Missing value for --backbone" >&2; exit 1; }
      configure_backbone "$2"
      shift 2
      ;;
    --backbone=*)
      configure_backbone "${1#*=}"
      shift
      ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)
      if [[ -n "${DATASET}" ]]; then
        EXTRA_ARGS+=("$1")
      else
        DATASET="$1"
      fi
      shift
      ;;
  esac
done

if [[ "${LIST_BACKBONES}" == 1 ]]; then
  list_backbones
  exit 0
fi

if [[ "${LIST_ONLY}" == 1 ]]; then
  list_datasets
  exit 0
fi

if [[ -z "${DATASET}" ]]; then
  usage >&2
  exit 1
fi

configure_backbone "${BACKBONE}"
DATA_SET_ARG=$(normalize_dataset_key "${DATASET}")
DATA_PATH="${DATASET_ROOT}/${DATASET}"
OUTPUT_DIR="${OUTPUT_ROOT}/${BACKBONE}/${DATASET}"
PRETRAIN_CKPT="${PRETRAIN_CKPT_VALUE:-${DEFAULT_CKPT}}"
PRETRAIN_CKPT=$(abspath_from_project "${PRETRAIN_CKPT}")
BATCH_SIZE="${BATCH_SIZE_VALUE:-${DEFAULT_BATCH_SIZE}}"
EPOCHS="${EPOCHS_VALUE:-${DEFAULT_EPOCHS}}"
LR="${LR_VALUE:-${DEFAULT_LR}}"

if [[ ! -f "${ENTRYPOINT_PATH}" ]]; then
  echo "Missing training entrypoint: ${ENTRYPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "Missing dataset split directory: ${DATA_PATH}" >&2
  exit 1
fi

CMD=(
  torchrun --nproc_per_node="${GPUS}" --module "${ENTRYPOINT_MODULE}"
  --model "${MODEL_NAME}"
  --data_set "${DATA_SET_ARG}"
  --data_path "${DATA_PATH}"
  --finetune "${PRETRAIN_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --num_frames "${NUM_FRAMES}"
  --sampling_rate "${SAMPLING_RATE}"
  "${BACKBONE_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'cd %q\n' "${TRAINING_DIR}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "Missing checkpoint: ${PRETRAIN_CKPT}" >&2
  echo "Set PRETRAIN_CKPT=/path/to/checkpoint or place the expected file under ${CKPT_ROOT}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${TRAINING_DIR}"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} "${CMD[@]}"
