# Installation

## Recommended Environment

- Linux with CUDA-capable GPUs
- Python 3.10
- PyTorch matching your CUDA driver
- `ffmpeg` available on `PATH`

## Conda Setup

```bash
conda create -n surgrec python=3.10 -y
conda activate surgrec
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Install PyTorch from the official selector for your CUDA version if the default wheel is not suitable.

`pip install -e .` puts `surgrec_video` on the import path so the entrypoints run from any working directory. Without it, run every command from the repository root.

## Optional VLM Dependencies

The `vlm_eval/` scripts require Hugging Face Transformers-compatible checkpoints and may need extra packages such as `accelerate`, `qwen-vl-utils`, and `av`, depending on the model backend.

## Checkpoints

```bash
hf download SichengLu/SurgRec --local-dir checkpoints
```

## Sanity Check

```bash
python -m compileall -q surgrec_video tools vlm_eval
python -m pytest tests -q
bash scripts/train/finetune_surgrec_video.sh --list-backbones
DATASET_ROOT=splits PRETRAIN_CKPT=/tmp/placeholder.pth GPUS=1 EPOCHS=1 \
  bash scripts/train/finetune_surgrec_video.sh cataract-1k-phase --backbone surgrec_mae --dry-run
```

The tests need only `numpy` and `pytest`; they do not touch the GPU stack.

The last command prints the `torchrun` invocation it would run without starting
training, so it works before any video data is in place.
