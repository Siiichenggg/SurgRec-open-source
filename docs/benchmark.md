# Paper Benchmark Reproduction

## Reported Results

The full per-dataset table (all baselines and both SurgRec variants, from
Table 2 of the [paper](https://arxiv.org/abs/2603.29966)) is in the
[Results section of the README](../README.md#results).

SurgRec-MAE is the best method on 14 of 16 datasets and leads the mean Acc@1 by
9.4 points over the strongest baseline. SurgRec-JEPA is close to the baseline
envelope on the mean; it is the single best method on Cholec80 and ties
SurgRec-MAE on M2CAI16. Cat-21 is saturated at one value across every method,
which makes that row uninformative.

## Datasets

`splits/` may contain auxiliary datasets such as PitVis, but the default benchmark scripts intentionally use only the paper matrix:

- AlxSuture
- AutoLaparo
- cat-21
- cataract-101
- cataract-1k-phase
- cholec80
- Colonoscopic-web
- hyper-kvasir
- JIGSAWS
- kvasir-capsule
- LapGyn_dataset
- LDPolyVideo
- M2CAI16-Workflow
- MultiBypass140
- SAR-RARP50
- SurgicalActions160

## SSL Models

```bash
DATASET_ROOT=data/splits_local CKPT_ROOT=checkpoints \
  bash scripts/benchmark/run_ssl_benchmark.sh --dry-run
```

Default SSL backbones are `dinov3`, `dinov3_surgenetxl`, `videomae`, `surgrec_mae`, `jepa_vitl16`, and `surgrec_jepa`.

## VLM Models

```bash
DATASET_ROOT=data/splits_local MODEL_ROOT=checkpoints \
  bash scripts/benchmark/run_vlm_benchmark.sh --dry-run
```

Default VLMs are `qwen3`, `llava-next`, and `qwen25`, each with `baseline` and `stability_v1` prompt variants.
