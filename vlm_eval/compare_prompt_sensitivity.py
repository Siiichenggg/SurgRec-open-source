#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelRun:
    model: str
    before_root: Path
    after_root: Path


def parse_model_spec(spec: str) -> ModelRun:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --model spec: {spec!r}. Expected format: model_name::before_root::after_root"
        )
    model, before_root, after_root = parts
    return ModelRun(model=model, before_root=Path(before_root), after_root=Path(after_root))


def list_datasets(dataset_root: Path) -> List[str]:
    names: List[str] = []
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and (item / "train.csv").is_file() and (item / "test.csv").is_file():
            names.append(item.name)
    return names


def load_meta(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric(meta: Optional[Dict], key: str, default=None):
    if not isinstance(meta, dict):
        return default
    return meta.get(key, default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs prompt-variant VLM results")
    parser.add_argument("--dataset-root", default="data/splits")
    parser.add_argument("--output-root", default="vlm_eval")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Format: model_name::before_root::after_root (repeat for multiple models)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = list_datasets(dataset_root)
    models = [parse_model_spec(spec) for spec in args.model]

    rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for model_run in models:
        model_rows = []
        before_weighted_correct = 0
        before_weighted_total = 0
        after_weighted_correct = 0
        after_weighted_total = 0

        for dataset in datasets:
            before_meta = load_meta(model_run.before_root / dataset / "meta.json")
            after_meta = load_meta(model_run.after_root / dataset / "meta.json")

            before_total = metric(before_meta, "total")
            before_accuracy = metric(before_meta, "accuracy")
            after_total = metric(after_meta, "total")
            after_accuracy = metric(after_meta, "accuracy")

            delta = None
            if isinstance(before_accuracy, (int, float)) and isinstance(after_accuracy, (int, float)):
                delta = float(after_accuracy) - float(before_accuracy)

            before_prompt_variant = metric(before_meta, "prompt_variant", "baseline") if before_meta else None
            after_prompt_variant = metric(after_meta, "prompt_variant", "baseline") if after_meta else None

            row = {
                "model": model_run.model,
                "dataset": dataset,
                "before_total": before_total,
                "before_accuracy": before_accuracy,
                "after_total": after_total,
                "after_accuracy": after_accuracy,
                "delta_accuracy": delta,
                "before_label_source": metric(before_meta, "label_source"),
                "after_label_source": metric(after_meta, "label_source"),
                "before_prompt_variant": before_prompt_variant,
                "after_prompt_variant": after_prompt_variant,
                "after_finished": bool(after_meta is not None),
            }
            rows.append(row)
            model_rows.append(row)

            if isinstance(before_total, int) and isinstance(before_accuracy, (int, float)):
                before_weighted_total += before_total
                before_weighted_correct += before_total * float(before_accuracy)
            if isinstance(after_total, int) and isinstance(after_accuracy, (int, float)):
                after_weighted_total += after_total
                after_weighted_correct += after_total * float(after_accuracy)

        completed = sum(1 for r in model_rows if r["after_finished"])
        total_datasets = len(model_rows)

        before_overall = (
            before_weighted_correct / before_weighted_total if before_weighted_total > 0 else None
        )
        after_overall = after_weighted_correct / after_weighted_total if after_weighted_total > 0 else None
        overall_delta = (
            (after_overall - before_overall)
            if isinstance(before_overall, float) and isinstance(after_overall, float)
            else None
        )

        summary_rows.append(
            {
                "model": model_run.model,
                "datasets_total": total_datasets,
                "datasets_after_completed": completed,
                "completion_ratio": completed / total_datasets if total_datasets else 0.0,
                "before_weighted_accuracy": before_overall,
                "after_weighted_accuracy": after_overall,
                "delta_weighted_accuracy": overall_delta,
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_root / f"prompt_sensitivity_compare_{ts}.csv"
    json_path = output_root / f"prompt_sensitivity_compare_{ts}.json"

    fieldnames = [
        "model",
        "dataset",
        "before_total",
        "before_accuracy",
        "after_total",
        "after_accuracy",
        "delta_accuracy",
        "before_label_source",
        "after_label_source",
        "before_prompt_variant",
        "after_prompt_variant",
        "after_finished",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset_root": str(dataset_root),
        "models": [
            {
                "model": m.model,
                "before_root": str(m.before_root),
                "after_root": str(m.after_root),
            }
            for m in models
        ],
        "summary": summary_rows,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WROTE_CSV={csv_path}")
    print(f"WROTE_JSON={json_path}")
    print("SUMMARY")
    for item in summary_rows:
        print(json.dumps(item, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
