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

# 8x RTX 3090 fine-tuning script for VideoMAE (JEPA v3 checkpoint)
OUTPUT_DIR="${OUTPUT_ROOT}/jepa_v3_ft"
DATA_PATH="${DATA_ROOT}"
EVAL_DATA_PATH="${DATA_ROOT}"
MODEL_PATH="${CKPT_ROOT}/surgrec_jepa.pt"  # released name; the run used jepa_v3_60w_100e_20w_130e.pt
LOG_DIR="${OUTPUT_ROOT}/log"

# Optional: set visible GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=1

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# NOTE: dataset uses max label 71 -> nb_classes 72
# NOTE: eval/test CSVs are under EVAL_DATA_PATH

cd "${PROJECT_ROOT}"

LOG_FILE="${LOG_DIR}/jepa_v3_finetune_8gpu_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node="${GPUS}" \
  --module surgrec_video.entrypoints.run_class_finetuning_videomae \
  --backbone_variant videomaev2 \
  --model vit_large_patch16_224 \
  --data_set all \
  --nb_classes 72 \
  --data_path "${DATA_PATH}" \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --finetune "${MODEL_PATH}" \
  --log_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 8 \
  --num_sample 1 \
  --input_size 224 \
  --short_side_size 224 \
  --save_ckpt_freq 10 \
  --num_frames 16 \
  --sampling_rate 4 \
  --opt adamw \
  --lr 5e-4 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --epochs 50 \
  2>&1 | tee "${LOG_FILE}"
