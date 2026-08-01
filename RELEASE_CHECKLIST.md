# Public Release Checklist

- [x] Replace anonymized title/authors/contact in `README.md`, `CITATION.cff`, and BibTeX.
- [x] Decide final repository license with coauthors/institution. Set to CC BY-NC 4.0, the only option compatible with the VideoMAE code this derives from. **Confirm with all coauthors and your institution.**
- [x] Verify third-party license compatibility for VideoMAE, VideoMAE V2, DINO, JEPA, LLaVA, Qwen, timm, and any copied utilities. Verified and documented in `NOTICE`; none prohibits non-commercial redistribution. Two distribution caveats noted (do not vendor the GPLv3 PyAV wheel; DINOv3 files, which are not shipped, would carry their own license).
- [x] Remove or regenerate any split whose source dataset license does not permit redistribution. Per-dataset licenses verified and tabulated in `DATASETS.md`. All 16 splits are retained: 13 are covered by open licenses, and the three without an open redistribution grant (AlxSuture CC BY-NC-ND, JIGSAWS IRB release, LDPolypVideo unlicensed) are redistributed with permission obtained from the dataset providers. Keep those permission records on file in case of a takedown query.
- [~] Upload checkpoints externally and update `MODEL_ZOO.md`. The two SurgRec backbones are uploaded and verified but currently private, pending transfer to the lab's Hugging Face organization; once public, add the download command and direct links. Baselines are third-party and linked, not redistributed.
- [x] Run a sensitive-string scan for private paths, passwords, tokens, API keys, and machine-specific user names before publishing.
- [x] Run a smoke fine-tuning command with `--dry-run` and one tiny dataset.
- [x] Confirm no raw videos, model weights, logs, TensorBoard files, caches, or conda directories are tracked.
- [x] Add the final paper PDF/arXiv link after acceptance or de-anonymization. Linked to [arXiv:2603.29966](https://arxiv.org/abs/2603.29966); update to the venue citation once published.
- [x] NumPy 2.x compatibility. The one removed alias (`numpy.lib.function_base.disp`) was dropped and the tree was scanned for other removed aliases (`np.float`, `np.int`, `np.bool`, `np.object`, `np.alltrue`, `np.trapz`, ...) — none remain, so `requirements.txt` leaves `numpy` unpinned.
