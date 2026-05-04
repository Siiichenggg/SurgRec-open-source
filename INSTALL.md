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
```

Install PyTorch from the official selector for your CUDA version if the default wheel is not suitable.

## Optional VLM Dependencies

The `vlm_eval/` scripts require Hugging Face Transformers-compatible checkpoints and may need extra packages such as `accelerate`, `qwen-vl-utils`, and `av`, depending on the model backend.

## Sanity Check

```bash
python -m py_compile surgrec_video/run_class_finetuning_videomaev2.py
python -m py_compile vlm_eval/run_qwen3_vlm_test.py
```
