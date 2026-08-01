# Paper Benchmark Reproduction

## Reported Results

The full per-dataset table (all baselines and both SurgRec variants, from
Table 2 of the [paper](https://arxiv.org/abs/2603.29966)) is in the
[Results section of the README](../README.md#results).

SurgRec-MAE is strictly the best of the seven compared methods on 11 of the 16
datasets and ties for best on 3 more (M2CAI16, SurgicalActions160, cat-21). It
does not lead on cataract-1k, where JEPA is highest, or on Cholec80, where
SurgRec-JEPA is. On the mean it is 9.4 points above the strongest other column
(the SR-MAE without-balanced-sampling ablation, 35.92) and 11.3 points above the
strongest third-party baseline (JEPA, 34.02).

Every method scores exactly 15.69 on cat-21, consistent with all of them
collapsing to a single class; we treat that row as uninformative rather than as
evidence of parity.

Two columns of the table cannot be rerun from this repository: the
SR-MAE (w/o bal.) ablation and DINOv3-SurgeNetXL, neither of which is a released
checkpoint.

## Datasets

`splits/` may contain auxiliary datasets such as PitVis, but the default benchmark scripts intentionally use only the paper matrix:

- AIxSuture
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
- LDPolypVideo
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
