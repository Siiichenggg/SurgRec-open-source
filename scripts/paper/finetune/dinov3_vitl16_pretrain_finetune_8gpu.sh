#!/usr/bin/env bash
# Paper configuration, released verbatim except for machine-specific paths.
#
#   CKPT_ROOT    directory holding the released pretraining checkpoints
#   DATA_ROOT    directory holding train.csv (and test.csv when evaluating)
#   TEST_ROOT    directory holding the held-out test.csv
#   OUTPUT_ROOT  directory for checkpoints and logs
#   GPUS         number of processes; the paper used 8
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory containing train.csv}"
TEST_ROOT="${TEST_ROOT:-${DATA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/paper}"
GPUS="${GPUS:-8}"

# 8x GPU fine-tuning + evaluation script for VideoMAE with DINOv3 ViT-L/16 pretrain
OUTPUT_DIR="${OUTPUT_ROOT}/dinov3_vitl16_pretrain_ft"
DATA_PATH="${DATA_ROOT}"
MODEL_PATH="${CKPT_ROOT}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
CKPT_PATH="${OUTPUT_ROOT}/dinov3_vitl16_pretrain_ft/checkpoint-49.pth"
LOG_DIR="${OUTPUT_ROOT}/log"

# Optional: set visible GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=1

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${PROJECT_ROOT}"

LOG_FILE="${LOG_DIR}/dinov3_vitl16_pretrain_finetune_eval_8gpu_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Stage 1/2: Finetune"
torchrun --nproc_per_node="${GPUS}" \
  --module surgrec_video.entrypoints.run_class_finetuning_videomae \
  --backbone_variant videomaev2 \
  --model vit_large_patch16_224 \
  --data_set all \
  --nb_classes 72 \
  --data_path "${DATA_PATH}" \
  --finetune "${MODEL_PATH}" \
  --log_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 4 \
  --num_sample 2 \
  --input_size 336 \
  --short_side_size 336 \
  --save_ckpt_freq 10 \
  --num_frames 16 \
  --sampling_rate 4 \
  --opt adamw \
  --lr 2e-3 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --epochs 50 \
  --disable_eval_during_finetuning \
  2>&1 | tee "${LOG_FILE}"

echo "[INFO] Stage 2/2: Eval"
if [[ ! -f "${DATA_PATH}/test.csv" ]]; then
  echo "[ERROR] Missing test.csv at ${DATA_PATH}/test.csv" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] Missing checkpoint at ${CKPT_PATH}" >&2
  exit 1
fi

torchrun --nproc_per_node="${GPUS}" \
  --module surgrec_video.entrypoints.run_class_finetuning_videomae \
  --backbone_variant videomaev2 \
  --model vit_large_patch16_224 \
  --data_set all \
  --nb_classes 72 \
  --data_path "${DATA_PATH}" \
  --finetune "${CKPT_PATH}" \
  --log_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 4 \
  --num_sample 2 \
  --input_size 336 \
  --short_side_size 336 \
  --num_frames 16 \
  --sampling_rate 4 \
  --eval \
  2>&1 | tee -a "${LOG_FILE}"
