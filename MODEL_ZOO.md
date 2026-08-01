# Model Zoo

Model weights are not stored in Git. The two SurgRec backbones are published on
the Hugging Face Hub at
[SichengLu/SurgRec](https://huggingface.co/SichengLu/SurgRec):

```bash
hf download SichengLu/SurgRec --local-dir checkpoints
```

The public scripts read `CKPT_ROOT` and `PRETRAIN_CKPT`; avoid hard-coding machine-local paths.

The general-domain baselines (DINOv3, VideoMAE, JEPA) are third-party and are not
redistributed here — obtain them from their original sources. Two columns of the
paper's Table 2 are neither released nor reproducible from this repository: the
SR-MAE (w/o bal.) ablation and DINOv3-SurgeNetXL, both of which require the
pretraining pipeline.

| Model | Backbone | Role in paper | Expected local filename | Size | Notes |
| --- | --- | --- | --- | --- | --- |
| SurgRec-MAE | ViT-B | Ours, masked video reconstruction | `checkpoints/surgrec_mae.pth` | 1.1 GB | tube mask 0.9, decoder depth 4, epoch 149. Load with `model_key=model`. |
| SurgRec-JEPA | ViT-L | Ours, latent predictive pretraining | `checkpoints/surgrec_jepa.pt` | 4.8 GB | epoch 131. Load with `model_key=encoder`. |
| VideoMAE baseline | ViT-B | General-domain SSL baseline | `checkpoints/videomae.pth` | 360 MB | Release alias for `videomae_b.pth`. |
| JEPA baseline | ViT-L | General-domain JEPA baseline | `checkpoints/jepa_vitl16.pth.tar` | 4.8 GB | Release alias for `vitl16.pth.tar`. |
| DINOv3 baseline | ViT-L/16 | General-domain DINOv3 baseline | `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 1.2 GB | Fine-tuned through the ViT video head. |
| DINOv3-SurgeNetXL | ViT-L/16 | Domain-adapted reference | `checkpoints/DINOv3_ViTl16_size336_SurgeNetXL.pth` | 1.2 GB | **Not third-party and not released.** Produced by continuing DINOv3 pretraining on our clinical subset (paper, Sec. 4.2), following the SurgeNetXL line of work. The continued-pretraining code is not part of this repository, so the `dinov3_surgenetxl` benchmark entry cannot be reproduced from this release. |

You can also point directly at a specific checkpoint file:

```bash
PRETRAIN_CKPT=/path/to/jepa_v3_60w_100e_20w_130e.pt \
  bash scripts/train/finetune_surgrec_video.sh AutoLaparo --backbone surgrec_jepa
```
