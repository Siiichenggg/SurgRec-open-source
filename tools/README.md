# Tools

The utility scripts are grouped by purpose:

- `tools/data/relocate_splits.py`: rewrite sanitized split CSVs to local absolute dataset paths.
- `tools/data/build_multibypass_iae_splits.py`: build the MultiBypass140 IAE subset splits.
- `tools/validation/check_and_fix_videos.py`: verify that split-listed videos can be decoded by `decord` and optionally repair them with `ffmpeg`.
- `tools/analysis/summarize_multibypass140_results.py`: aggregate fold-level MultiBypass140 metrics.
- `tools/analysis/extract_fold_results.py`: pull per-fold final metrics out of a combined train/test log.

Examples:

```bash
python tools/data/relocate_splits.py   --input-root splits   --output-root data/splits_local   --data-root /path/to/your/dataset_root
```

```bash
python tools/validation/check_and_fix_videos.py   --csv data/splits_local/cholec80/test.csv   --check
```

```bash
python tools/validation/check_and_fix_videos.py   --csv data/splits_local/cholec80/test.csv   --fix --strategy both
```

```bash
python tools/analysis/extract_fold_results.py   --log outputs/multibypass140/allfolds_train_test.log   --fold 0 --format csv --out fold0_results.csv
```
