#!/usr/bin/env python
import argparse
import pickle
import re
from pathlib import Path

CASE_RE = re.compile(r"/([A-Z]{3}\d{2})/")


def extract_case_id(path: str) -> str | None:
    match = CASE_RE.search(path)
    if match:
        return match.group(1)
    name = Path(path).name
    if name.startswith("BBP"):
        return name.split("_")[0]
    return None


def load_case_ids(pickle_path: Path) -> set[str]:
    if not pickle_path.is_file():
        return set()
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return set(data.keys())
    return set()


def collect_case_ids(labels_root: Path, split: str, fold: int) -> set[str]:
    ids: set[str] = set()
    for center in ["bern", "strasbourg"]:
        p = labels_root / center / "labels" / split
        if split == "train":
            name = f"1fps_100_{fold}_with_iae.pickle"
        else:
            name = f"1fps_{fold}_with_iae.pickle"
        ids |= load_case_ids(p / name)
    return ids


def filter_csv(src: Path, dst: Path, case_ids: set[str]) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            case_id = extract_case_id(parts[0])
            if case_id and case_id in case_ids:
                fout.write(line)
                kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["phase", "step"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_root = dataset_root / "datasets" / "MultiBypass140" / "multibypass06" / "labels"
    if not labels_root.exists():
        raise SystemExit(f"Labels root not found: {labels_root}")

    split_base = dataset_root / "csv_splits" / args.task
    out_base = dataset_root / "iae_splits" / args.task

    for split in ["train", "val", "test"]:
        src = split_base / split / f"fold{args.fold}.csv"
        if not src.exists():
            raise SystemExit(f"Missing split CSV: {src}")
        ids = collect_case_ids(labels_root, split, args.fold)
        if not ids:
            raise SystemExit(f"No case IDs for split={split} fold={args.fold}")
        dst = out_base / split / f"fold{args.fold}.csv"
        kept = filter_csv(src, dst, ids)
        print(f"{split} fold{args.fold}: kept {kept} rows -> {dst}")


if __name__ == "__main__":
    main()
