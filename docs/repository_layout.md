# Repository Layout

The public repository is organized around the supported user workflows.

```text
surgrec_video/   Python package for video fine-tuning and pretraining
scripts/         Shell entrypoints for training and VLM evaluation
splits/          Lightweight train/val/test split metadata
tools/           Data preparation, validation, and analysis helpers
vlm_eval/        VLM baseline evaluation code
configs/         Model registry and checkpoint placeholders
docs/            Additional release notes
```

Historical experiment forks, generated outputs, local checkpoints, TensorBoard logs, and machine-specific launchers are intentionally excluded from the release tree.
