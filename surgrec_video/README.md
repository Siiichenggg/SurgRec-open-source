# surgrec_video

`surgrec_video` is the training package used by the public release.

## Internal Layout

- `entrypoints/`: runnable Python module entrypoints for downstream fine-tuning.
- `models/`: backbone definitions and checkpoint-compatible model builders.
- `data/`: dataset builders and split readers.
- `engine/`: training and evaluation loops.
- `core/`: optimizer, logging, distributed, and checkpoint utilities.
- `augment/`: mixup, transforms, random erasing, and video augmentation helpers.

## Public Interface

Use the shell launchers under `scripts/` for normal usage.

- Train: `scripts/train/finetune_surgrec_video.sh`
- Evaluate VLM baselines: `scripts/eval/evaluate_vlm.sh`

Direct module entrypoints are available for debugging and smoke tests:

- `python -m surgrec_video.entrypoints.run_class_finetuning_videomae ...`
- `python -m surgrec_video.entrypoints.run_class_finetuning_medical ...`
