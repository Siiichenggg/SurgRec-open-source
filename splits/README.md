# Dataset Splits

This directory contains release metadata for downstream evaluation splits. Each dataset directory is intentionally small and should contain only:

- `train.csv`
- `val.csv`
- `test.csv`
- optional `label_map.json` or `label_mapping.json` for class names
- optional README or license notes from the source dataset

CSV rows use the format:

```text
/path/to/video label_id
```

Raw videos are not included. Use `tools/data/relocate_splits.py` to rewrite paths for your local dataset root before training.
