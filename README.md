# SurgRec: Scaling Video Pretraining for Surgical Foundation Models

This repository contains the public release scaffold for **SurgRec**, a reproducible recipe for surgical video pretraining, downstream fine-tuning, and VLM baseline evaluation.

> Paper: *Scaling Video Pretraining for Surgical Foundation Models*  
> Status: pre-release cleanup. Replace anonymized paper metadata, model URLs, and dataset links before publishing.

## Highlights

- Surgical video pretraining and fine-tuning code with VideoMAE-style entrypoints.
- Unified downstream training launcher for paper models (`surgrec_mae`, `surgrec_jepa`) and baselines (`videomae`, `jepa_vitl16`, `dinov3`, `dinov3_surgenetxl`).
- Sanitized train/val/test split metadata for the 16-dataset paper benchmark plus auxiliary splits.
- VLM baseline evaluation scripts for Qwen3-VL, Qwen2.5-VL, and LLaVA-NeXT.
- Release docs, environment files, citation metadata, and model registry placeholders.

## Repository Layout

```text
surgrec_video/   Python package for video fine-tuning and pretraining
scripts/         Public shell entrypoints for training and VLM evaluation
splits/          Lightweight downstream split metadata; no raw videos
tools/           Data preparation, validation, and analysis helpers
vlm_eval/        VLM baseline evaluation code
configs/         Model registry and checkpoint placeholders
docs/            Additional release notes
```

## Quick Start

```bash
conda create -n surgrec python=3.10 -y
conda activate surgrec
pip install -r requirements.txt
```

Prepare dataset CSVs with local absolute video paths:

```bash
python tools/data/relocate_splits.py   --input-root splits   --output-root data/splits_local   --data-root /path/to/your/dataset_root
```

List available downstream datasets and supported backbones:

```bash
DATASET_ROOT=data/splits_local bash scripts/train/finetune_surgrec_video.sh --list
bash scripts/train/finetune_surgrec_video.sh --list-backbones
```

Fine-tune a checkpoint on one downstream dataset:

```bash
DATASET_ROOT=data/splits_local CKPT_ROOT=/path/to/checkpoints GPUS=8 bash scripts/train/finetune_surgrec_video.sh cholec80 --backbone surgrec_mae
```


Run the paper benchmark launchers in dry-run mode before starting long jobs:

```bash
DATASET_ROOT=data/splits_local bash scripts/benchmark/run_ssl_benchmark.sh --dry-run
DATASET_ROOT=data/splits_local bash scripts/benchmark/run_vlm_benchmark.sh --dry-run
```

Run a VLM baseline evaluation:

```bash
DATASET_ROOT=data/splits_local MODEL_ROOT=/path/to/vlm_checkpoints bash scripts/eval/evaluate_vlm.sh cholec80 qwen3
```

## What Is Not Included

- Raw clinical or web videos.
- Model checkpoints or third-party VLM weights.
- TensorBoard files, generated outputs, caches, or conda environments.
- Historical experiment forks and machine-specific launchers.
- Private/internal datasets unless their licenses permit redistribution.

## Release Notes For Maintainers

Before public release, complete `RELEASE_CHECKLIST.md`, upload model weights to a durable host, replace all `TODO` placeholders, and verify every redistributed dataset split complies with the source dataset license.

## Acknowledgements

This project builds on VideoMAE, VideoMAE V2, timm, PyTorch, decord, and related open-source surgical-video and VLM tooling. Third-party components retain their original licenses; see `NOTICE`.

## Citation

```bibtex
@inproceedings{surgrec2026,
  title = {Scaling Video Pretraining for Surgical Foundation Models},
  author = {TODO: add authors after de-anonymization},
  booktitle = {TODO: add venue},
  year = {2026}
}
```
