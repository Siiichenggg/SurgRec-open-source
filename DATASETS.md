# Dataset And Split Preparation

This repository releases split metadata and code only. It does **not** redistribute videos.

## Split Format

Each dataset directory under `splits/` contains the minimal metadata needed for training and evaluation:

```text
splits/<dataset>/
  train.csv
  val.csv
  test.csv
  label_map.json        # optional
  README.md             # optional
  LICENSE.txt           # optional
```

Each CSV uses the common VideoMAE-style format:

```text
/path/to/video.mp4 label_id
```

The committed CSVs use `__DATA_ROOT__` instead of a machine-specific absolute path. Convert them before training:

```bash
python tools/data/relocate_splits.py   --input-root splits   --output-root data/splits_local   --data-root /path/to/your/dataset_root
```

## Expected Layout After Relocation

```text
data/splits_local/
  cholec80/
    train.csv
    val.csv
    test.csv
  M2CAI16-Workflow/
    train.csv
    val.csv
    test.csv
```

The video files should be arranged under your `--data-root` using the same relative paths as the original split metadata.

## License And Privacy

Use every dataset according to its original license and terms. Do not redistribute protected health information or private clinical videos. If a source dataset does not permit redistribution of splits, remove that split before publishing.
