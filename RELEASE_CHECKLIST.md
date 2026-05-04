# Public Release Checklist

- [ ] Replace anonymized title/authors/contact in `README.md`, `CITATION.cff`, and BibTeX.
- [ ] Decide final repository license with coauthors/institution. Current scaffold uses a non-commercial research-use notice because parts derive from VideoMAE-style code.
- [ ] Verify third-party license compatibility for VideoMAE, VideoMAE V2, DINO, JEPA, LLaVA, Qwen, timm, and any copied utilities.
- [ ] Remove or regenerate any split whose source dataset license does not permit redistribution.
- [ ] Upload checkpoints externally and update `MODEL_ZOO.md`.
- [ ] Run a sensitive-string scan for private paths, passwords, tokens, API keys, and machine-specific user names before publishing.
- [ ] Run a smoke fine-tuning command with `--dry-run` and one tiny dataset.
- [ ] Confirm no raw videos, model weights, logs, TensorBoard files, caches, or conda directories are tracked.
- [ ] Add the final paper PDF/arXiv link after acceptance or de-anonymization.
