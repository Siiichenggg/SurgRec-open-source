#!/usr/bin/env python
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev, median

MODEL_TAGS = {
    "dinov3_vitl16_pretrain": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd",
    "videomae_e149": "videomae_e149",
    "videomae_b": "videomae_b",
    "jepa_v3_60w": "jepa_v3_60w_100e_20w_130e",
    "jepa_vitl16": "jepa",
    "DINOv3_ViTl16_SurgeNetXL": "DINOv3_ViTl16_size336_SurgeNetXL",
}

MODEL_LIST = list(MODEL_TAGS.keys())
TASKS = ["phase", "step"]
FOLDS = [0, 1, 2, 3, 4]


def parse_allfolds_metrics(output_root: Path):
    metrics = {}
    allfolds_logs = list(output_root.glob("multibypass140/*/allfolds_*.out"))
    for log_path in allfolds_logs:
        if not log_path.exists():
            continue
        current_task = None
        current_fold = None
        current_model = None
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("[INFO] Task=") and "Fold=" in line:
                parts = line.replace("[INFO] ", "").split()
                task_part = next((p for p in parts if p.startswith("Task=")), None)
                fold_part = next((p for p in parts if p.startswith("Fold=")), None)
                if task_part and fold_part:
                    current_task = task_part.split("=", 1)[1]
                    try:
                        current_fold = int(fold_part.split("=", 1)[1])
                    except ValueError:
                        current_fold = None
                continue

            if line.startswith("[INFO] Test:") and "/ MultiBypass140" in line:
                current_model = line.split("[INFO] Test:", 1)[1].split("/", 1)[0].strip()
                continue

            if "Accuracy of the network on" in line and "Top-1" in line and "Top-5" in line:
                if current_task is None or current_fold is None or current_model is None:
                    continue
                try:
                    top1 = float(line.split("Top-1:", 1)[1].split("%", 1)[0].strip())
                    top5 = float(line.split("Top-5:", 1)[1].split("%", 1)[0].strip())
                except Exception:
                    continue
                metrics[(current_task, current_fold, current_model)] = (top1, top5)
    return metrics


def parse_log_metrics(log_path: Path):
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "Final top-1" in obj:
            best = (obj.get("Final top-1"), obj.get("Final Top-5"))
    return best


def load_video_ids(test_csv: Path):
    mapping = {}
    with test_csv.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            path = parts[0]
            vid = Path(path).stem
            center = "bern" if "/bern/" in path else ("strasbourg" if "/strasbourg/" in path else "unknown")
            mapping[vid] = center
    return mapping


def compute_center_metrics(output_dir: Path, test_csv: Path):
    mapping = load_video_ids(test_csv)
    txt_files = sorted(output_dir.glob("*.txt"))
    if not txt_files:
        return {}
    rows = []
    for txt in txt_files:
        for line in txt.read_text(encoding="utf-8").splitlines()[1:]:
            if not line:
                continue
            if "[" not in line or "]" not in line:
                continue
            name = line.split("[", 1)[0].strip()
            if not name:
                continue
            logits = line.split("[", 1)[1].split("]", 1)[0]
            scores = [float(x) for x in logits.split(",") if x.strip()]
            if not scores:
                continue
            tail = line.split("]", 1)[1].strip()
            tokens = tail.split()
            if not tokens:
                continue
            label_token = tokens[1] if len(tokens) > 1 else tokens[0]
            try:
                label = int(label_token)
            except ValueError:
                continue
            pred = max(range(len(scores)), key=lambda i: scores[i])
            rows.append((name, pred, label))

    by_center = {"bern": [], "strasbourg": []}
    for vid, pred, label in rows:
        center = mapping.get(vid, "unknown")
        if center in by_center:
            by_center[center].append(1.0 if pred == label else 0.0)

    results = {}
    for center, vals in by_center.items():
        if vals:
            results[center] = sum(vals) / len(vals) * 100.0
    return results


def summarize(values):
    if not values:
        return None
    m = mean(values)
    sd = pstdev(values)
    med = median(values)
    return {
        "mean": m,
        "std": sd,
        "median": med,
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/finetune")
    parser.add_argument("--dataset-root", default="data/splits/MultiBypass140")
    parser.add_argument("--use-iae", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)

    table_a = []
    table_b = []
    table_c = []

    allfolds_metrics = parse_allfolds_metrics(output_root)

    for task in TASKS:
        for model in MODEL_LIST:
            fold_metrics = []
            for fold in FOLDS:
                output_dir = output_root / "multibypass140" / task / f"fold{fold}" / f"finetune_{model}" / "MultiBypass140" / MODEL_TAGS[model]
                log_path = output_dir / "log.txt"
                metrics = parse_log_metrics(log_path)
                if metrics is None:
                    metrics = allfolds_metrics.get((task, fold, model))
                if metrics is None:
                    table_a.append((task, fold, model, None, None))
                    continue
                top1, top5 = metrics
                fold_metrics.append(top1)
                table_a.append((task, fold, model, top1, top5))

                split_base = "iae_splits" if args.use_iae else "csv_splits"
                test_csv = dataset_root / split_base / task / "test" / f"fold{fold}.csv"
                center_metrics = compute_center_metrics(output_dir, test_csv)
                if center_metrics:
                    for center, acc in center_metrics.items():
                        table_c.append((task, fold, model, center, acc))

            summary = summarize(fold_metrics)
            if summary:
                table_b.append((task, model, summary))

    # Print Table A
    print("\nTable A: per-fold metrics")
    print("task,fold,model,top1,top5")
    for row in table_a:
        task, fold, model, top1, top5 = row
        print(f"{task},{fold},{model},{'' if top1 is None else round(top1,3)},{'' if top5 is None else round(top5,3)}")

    # Print Table B
    print("\nTable B: cross-fold summary (mean±std, median[min,max])")
    print("task,model,mean,std,median,min,max")
    for task, model, summary in table_b:
        print(f"{task},{model},{summary['mean']:.3f},{summary['std']:.3f},{summary['median']:.3f},{summary['min']:.3f},{summary['max']:.3f}")

    # Print Table C
    if table_c:
        print("\nTable C: center-wise (Bern/Strasbourg)")
        print("task,fold,model,center,top1")
        for row in table_c:
            task, fold, model, center, acc = row
            print(f"{task},{fold},{model},{center},{acc:.3f}")


if __name__ == "__main__":
    main()
