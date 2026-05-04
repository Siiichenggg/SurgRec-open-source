# Model Zoo

Model weights are intentionally not stored in Git. Upload release checkpoints to a durable model host, then replace the placeholders below. The public scripts read `CKPT_ROOT` and `PRETRAIN_CKPT`; avoid hard-coding machine-local paths.

| Model | Backbone | Role in paper | Expected local filename | Notes |
| --- | --- | --- | --- | --- |
| SurgRec-MAE | ViT-B | Ours, balanced masked video reconstruction | `checkpoints/surgrec_mae.pth` | Release alias for the balanced MAE checkpoint. |
| SurgRec-JEPA | ViT-L | Ours, latent predictive pretraining | `checkpoints/surgrec_jepa.pt` | Release alias for `jepa_v3_60w_100e_20w_130e.pt`. |
| VideoMAE baseline | ViT-B | General-domain SSL baseline | `checkpoints/videomae.pth` | General VideoMAE comparison. |
| JEPA baseline | ViT-L | General-domain JEPA baseline | `checkpoints/jepa_vitl16.pth.tar` | Generic JEPA comparison. |
| DINOv3 baseline | ViT-L/16 | General-domain DINOv3 baseline | `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Fine-tuned through the ViT video head. |
| DINOv3-SurgeNetXL | ViT-L/16 | Domain-adapted DINOv3 baseline | `checkpoints/DINOv3_ViTl16_size336_SurgeNetXL.pth` | Fine-tuned through the ViT video head. |

For local testing before release, you can point directly at an existing checkpoint:

```bash
PRETRAIN_CKPT=/path/to/jepa_v3_60w_100e_20w_130e.pt \
  bash scripts/train/finetune_surgrec_video.sh AutoLaparo --backbone surgrec_jepa
```
