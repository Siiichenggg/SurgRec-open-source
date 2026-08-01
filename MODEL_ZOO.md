# Model Zoo

Model weights are not stored in Git. The two SurgRec backbones are published on
the Hugging Face Hub at
[SichengLu/SurgRec](https://huggingface.co/SichengLu/SurgRec):

```bash
hf download SichengLu/SurgRec --local-dir checkpoints
```

The public scripts read `CKPT_ROOT` and `PRETRAIN_CKPT`; avoid hard-coding machine-local paths.

Baseline checkpoints are third-party and are not redistributed here — obtain them from their original sources.

| Model | Backbone | Role in paper | Expected local filename | Size | Notes |
| --- | --- | --- | --- | --- | --- |
| SurgRec-MAE | ViT-B | Ours, masked video reconstruction | `checkpoints/surgrec_mae.pth` | 1.1 GB | tube mask 0.9, decoder depth 4, epoch 149. Load with `model_key=model`. |
| SurgRec-JEPA | ViT-L | Ours, latent predictive pretraining | `checkpoints/surgrec_jepa.pt` | 4.8 GB | epoch 131. Load with `model_key=encoder`. |
| VideoMAE baseline | ViT-B | General-domain SSL baseline | `checkpoints/videomae.pth` | 360 MB | Release alias for `videomae_b.pth`. |
| JEPA baseline | ViT-L | General-domain JEPA baseline | `checkpoints/jepa_vitl16.pth.tar` | 4.8 GB | Release alias for `vitl16.pth.tar`. |
| DINOv3 baseline | ViT-L/16 | General-domain DINOv3 baseline | `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 1.2 GB | Fine-tuned through the ViT video head. |
| DINOv3-SurgeNetXL | ViT-L/16 | Domain-adapted DINOv3 baseline | `checkpoints/DINOv3_ViTl16_size336_SurgeNetXL.pth` | 1.2 GB | Fine-tuned through the ViT video head. |

For local testing before release, you can point directly at an existing checkpoint:

```bash
PRETRAIN_CKPT=/path/to/jepa_v3_60w_100e_20w_130e.pt \
  bash scripts/train/finetune_surgrec_video.sh AutoLaparo --backbone surgrec_jepa
```
