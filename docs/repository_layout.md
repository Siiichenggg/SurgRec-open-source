# Repository Layout

The public repository is organized around the supported user workflows.

```text
surgrec_video/   Python package for downstream video fine-tuning
scripts/train/   General-purpose fine-tuning launcher
scripts/paper/   The paper's own fine-tuning and test configurations
scripts/eval/    VLM evaluation entrypoints
scripts/benchmark/  Benchmark sweep launchers
splits/          Lightweight train/val/test split metadata
tools/           Data preparation, validation, and analysis helpers
vlm_eval/        VLM baseline evaluation code
configs/         Model registry mapping backbones to checkpoint filenames
docs/            Additional release notes
```

This release covers downstream fine-tuning, the benchmark, and VLM evaluation, and uses the released pretrained backbones. Generated outputs, local checkpoints, and TensorBoard logs are not tracked. The launchers under `scripts/paper/` are the ones actually used for the paper, rewritten so they no longer depend on the machine they ran on.
