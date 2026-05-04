# Fine-tuning Guide

The public release uses shell entrypoints under `scripts/`. Model-specific launchers and machine-specific paths are intentionally excluded from the release tree.

## Recommended Entrypoint

Use the unified launcher:

```bash
bash scripts/train/finetune_surgrec_video.sh <dataset-name> --backbone <surgrec_mae|surgrec_jepa|videomae|jepa_vitl16|dinov3|dinov3_surgenetxl>
```

Common options:

```bash
DATASET_ROOT=data/splits_local
CKPT_ROOT=/path/to/checkpoints
PRETRAIN_CKPT=/path/to/override_checkpoint.pth
OUTPUT_ROOT=outputs/finetune
GPUS=8
BATCH_SIZE=2
EPOCHS=50
LR=5e-4
```

Examples:

```bash
DATASET_ROOT=data/splits_local CKPT_ROOT=/path/to/checkpoints   bash scripts/train/finetune_surgrec_video.sh cholec80 --backbone surgrec_mae

DATASET_ROOT=data/splits_local CKPT_ROOT=/path/to/checkpoints   bash scripts/train/finetune_surgrec_video.sh AutoLaparo --backbone surgrec_jepa
```

## Advanced Usage

If you need full control, call the packaged entrypoints under `surgrec_video/entrypoints/` directly. `run_class_finetuning_videomae.py` covers SurgRec-MAE, SurgRec-JEPA, VideoMAE, JEPA, and DINOv3-style ViT checkpoints through the same ViT video head. `run_class_finetuning_medical.py` is retained for extra baselines (`dino`, `endofm`, `mocov2`).

```bash
python -m surgrec_video.entrypoints.run_class_finetuning_videomae --backbone_variant videomaev2 --help
python -m surgrec_video.entrypoints.run_class_finetuning_medical --backbone_variant dino --help
```
