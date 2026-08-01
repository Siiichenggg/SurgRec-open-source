# Paper Configurations

These are the launch scripts used to produce the paper's fine-tuning and test
runs, released with their hyperparameters unchanged. Only machine-specific
paths and the entrypoint invocation were rewritten: the original scripts ran a
flat copy of the training code, whereas the release ships it as the
`surgrec_video` package.

`scripts/train/finetune_surgrec_video.sh` is the general-purpose launcher and is
the right entry point for new experiments. Use the scripts here when you want
the paper's exact settings.

## Running

```bash
CKPT_ROOT=/path/to/checkpoints \
DATA_ROOT=/path/to/split_with_train_csv \
TEST_ROOT=/path/to/split_with_test_csv \
OUTPUT_ROOT=outputs/paper \
GPUS=8 \
  bash scripts/paper/finetune/videomae_e149_finetune_8gpu.sh
```

`DATA_ROOT` is required. `TEST_ROOT` defaults to `DATA_ROOT`, `CKPT_ROOT` to
`checkpoints/`, `OUTPUT_ROOT` to `outputs/paper/`, and `GPUS` to 8.

## Fine-tuning runs

All runs use `--data_set all --nb_classes 72`, 16 frames, sampling rate 4,
AdamW with betas `0.9 0.999` and weight decay `0.05`.

| Script | Backbone | Checkpoint | Batch | Samples | Input | LR | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `videomae_e149_finetune_8gpu.sh` | ViT-B/16 | `surgrec_mae.pth` | 8 | 1 | 224 | 5e-4 | 50 |
| `jepa_v3_finetune_8gpu.sh` | ViT-L/16 | `surgrec_jepa.pt` | 8 | 1 | 224 | 5e-4 | 50 |
| `videomae_base_finetune_8gpu.sh` | ViT-B/16 | `videomae_b.pth` | 8 | 1 | 224 | 5e-4 | 50 |
| `dinov3_vitl16_pretrain_finetune_8gpu.sh` | ViT-L/16 | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 4 | 2 | 336 | 2e-3 | 50 |
| `dinov3surgical_finetune_8gpu.sh` | ViT-L/16 | `DINOv3_ViTl16_size336_SurgeNetXL.pth` | 4 | 2 | 336 | 2e-3 | 50 |
| `videomae_e149_resume_8gpu.sh` | ViT-B/16 | `surgrec_mae.pth` | 8 | 1 | 224 | 5e-4 | 21 |

The two DINOv3 runs pass `--disable_eval_during_finetuning` and are evaluated
afterwards: `dinov3_vitl16_pretrain_finetune_8gpu.sh` runs fine-tuning and
evaluation in one script, and `dinov3surgical_finetune_8gpu.sh` pairs with
`test/dinov3surgical_test_8gpu.sh`. `videomae_e149_resume_8gpu.sh` resumes an
interrupted run for a further 21 epochs and is not a standalone configuration.

## Test runs

`test/dinov3surgical_test_8gpu.sh` evaluates `checkpoint-49.pth` from the
corresponding fine-tuning output directory with `--eval`.

## Notes on the released checkpoint names

- `videomae_e149_finetune_8gpu.sh` and `videomae_e149_resume_8gpu.sh` read
  `surgrec_mae.pth`, and `jepa_v3_finetune_8gpu.sh` reads `surgrec_jepa.pt`.
  These are the model-zoo release names for the checkpoints the runs called
  `videomae_e149.pth` and `jepa_v3_60w_100e_20w_130e.pt`.
- `videomae_base_finetune_8gpu.sh` originally read `checkpoint.pth`. No file by
  that name is part of the release; the general VideoMAE baseline ships as
  `videomae_b.pth`, which is what the script now points at.
