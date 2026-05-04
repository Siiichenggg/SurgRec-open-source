# VLM Evaluation

This directory contains zero-shot VLM baselines used in the paper. The release scripts evaluate one representative frame per clip against the same test splits used by the SSL fine-tuning benchmark.

Supported model kinds through `scripts/eval/evaluate_vlm.sh`:

- `qwen3`: Qwen3-VL-8B-Instruct
- `qwen25` / `qwen2.5`: Qwen2.5-VL-7B-Instruct
- `llava-next`: LLaVA-NeXT-vicuna-7b

## Single Dataset

```bash
DATASET_ROOT=data/splits_local MODEL_ROOT=checkpoints \
  bash scripts/eval/evaluate_vlm.sh cholec80 qwen3 --prompt-variant baseline
```

## Paper Matrix

```bash
DATASET_ROOT=data/splits_local MODEL_ROOT=checkpoints \
  bash scripts/benchmark/run_vlm_benchmark.sh --dry-run
```

The benchmark launcher runs the 16 paper datasets for `qwen3`, `qwen25`, and `llava-next` with both `baseline` and `stability_v1` prompt variants.

## Prompt Sensitivity

Use `compare_prompt_sensitivity.py` to compare two output directories from different prompt variants:

```bash
python vlm_eval/compare_prompt_sensitivity.py \
  --before outputs/paper_vlm_benchmark/runs/qwen3 \
  --after outputs/paper_vlm_benchmark/runs/qwen3 \
  --output-root outputs/paper_vlm_benchmark/prompt_compare
```
