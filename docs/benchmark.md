# Paper Benchmark Reproduction

The paper reports a 16-dataset downstream benchmark. `splits/` may contain auxiliary datasets such as PitVis, but the default benchmark scripts intentionally use only the paper matrix:

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
