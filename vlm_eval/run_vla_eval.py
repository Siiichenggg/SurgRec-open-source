#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_DATASET_ROOT = "data/splits"
DEFAULT_TRAINING_DIR = "surgrec_video"
DEFAULT_FINETUNE_OUTPUT_ROOT = "outputs/finetune"
DEFAULT_OUTPUT_ROOT = "vlm_eval/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal VLA evaluation runner")
    parser.add_argument("--config", default="vlm_eval/vla_eval_min.json", help="Path to config JSON")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Root dir with dataset splits")
    parser.add_argument("--training-dir", default=DEFAULT_TRAINING_DIR, help="Path to videomae_2")
    parser.add_argument("--finetune-output-root", default=DEFAULT_FINETUNE_OUTPUT_ROOT, help="Root dir for finetune outputs")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root dir for VLA eval outputs")
    parser.add_argument("--all-datasets", action="store_true", help="Auto-discover datasets under dataset-root")
    parser.add_argument("--use-finetune-output", action="store_true", default=True, help="Use finetune output dirs for resume checkpoints")
    parser.add_argument("--eval-pretrain", action="store_true", help="Fallback to pretrain weights when resume checkpoint missing")
    parser.add_argument("--gpus", type=int, default=1, help="torchrun nproc_per_node")
    parser.add_argument("--master-port", type=int, default=29502, help="torchrun master port")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--sampling-rate", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--test-num-segment", type=int, default=1)
    parser.add_argument("--test-num-crop", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None, help="Override CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--progress-file", default=None, help="Progress JSON path (default: <output-root>/progress.json)")
    parser.add_argument("--resume-progress", action="store_true", default=True, help="Skip items already marked ok in progress file")
    parser.add_argument("--no-resume-progress", action="store_false", dest="resume_progress")
    parser.add_argument("--force", action="store_true", help="Force re-run even if progress says ok")
    return parser.parse_args()


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_datasets(dataset_root: str) -> List[str]:
    if not os.path.isdir(dataset_root):
        return []
    names = []
    for entry in sorted(os.listdir(dataset_root)):
        d = os.path.join(dataset_root, entry)
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "train.csv")) and os.path.isfile(os.path.join(d, "test.csv")):
            names.append(entry)
    return names


def infer_nb_classes(train_csv: str) -> int:
    if not os.path.isfile(train_csv):
        return 0
    labels = set()
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            labels.add(row[-1])
    return len(labels)


def resolve_ckpt(output_root: str) -> Optional[str]:
    best = os.path.join(output_root, "checkpoint-best.pth")
    if os.path.isfile(best):
        return best
    pattern = re.compile(r"checkpoint-.*\.pth")
    candidates = []
    if os.path.isdir(output_root):
        for name in os.listdir(output_root):
            if pattern.match(name):
                candidates.append(os.path.join(output_root, name))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    last = os.path.join(output_root, "checkpoint.pth")
    if os.path.isfile(last):
        return last
    return None


def run_eval(
    training_dir: str,
    dataset: str,
    dataset_path: str,
    output_dir: str,
    resume_ckpt: Optional[str],
    pretrain_ckpt: str,
    model_name: str,
    nb_classes: int,
    args: argparse.Namespace,
    extra_flag: Optional[str],
) -> int:
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        os.path.join(training_dir, "run_class_finetuning_videomaev2.py"),
        "--eval",
        "--data_set",
        dataset,
        "--data_path",
        dataset_path,
        "--output_dir",
        output_dir,
        "--finetune",
        pretrain_ckpt,
        "--model",
        model_name,
        "--input_size",
        str(args.input_size),
        "--num_frames",
        str(args.num_frames),
        "--sampling_rate",
        str(args.sampling_rate),
        "--test_num_segment",
        str(args.test_num_segment),
        "--test_num_crop",
        str(args.test_num_crop),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--no_pin_mem",
        "--nb_classes",
        str(nb_classes),
    ]
    if resume_ckpt:
        cmd += ["--resume", resume_ckpt]
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("[INFO]", " ".join(cmd))
    if args.dry_run:
        return 0
    result = subprocess.run(cmd, env=env)
    return result.returncode


def parse_log_metrics(output_dir: str) -> Tuple[Optional[float], Optional[float]]:
    log_path = os.path.join(output_dir, "log.txt")
    if not os.path.isfile(log_path):
        return None, None
    last_json = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                last_json = line
    if not last_json:
        return None, None
    try:
        data = json.loads(last_json)
        top1 = data.get("Final top-1") or data.get("Final top-1", None)
        top5 = data.get("Final Top-5") or data.get("Final top-5", None)
        return top1, top5
    except Exception:
        return None, None


def load_progress(path: str) -> Dict[str, Dict]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_progress(path: str, progress: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)

    datasets = config.get("datasets", [])
    if args.all_datasets or not datasets:
        datasets = list_datasets(args.dataset_root)
    if not datasets:
        print("[ERROR] No datasets found.")
        return 1

    models = config.get("models", [])
    if not models:
        print("[ERROR] No models defined in config.")
        return 1

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    progress_file = args.progress_file or str(output_root / "progress.json")
    progress = load_progress(progress_file)

    summary_rows = []
    for dataset in datasets:
        dataset_path = os.path.join(args.dataset_root, dataset)
        train_csv = os.path.join(dataset_path, "train.csv")
        nb_classes = infer_nb_classes(train_csv)
        if nb_classes <= 0:
            nb_classes = 3

        for model in models:
            model_name = model["model_name"]
            pretrain_ckpt = model["pretrain_ckpt"]
            extra_flag = model.get("extra_flag")
            script_tag = model.get("script_tag", model.get("name", "model"))
            ckpt_tag = os.path.basename(pretrain_ckpt)
            ckpt_tag = os.path.splitext(ckpt_tag)[0]

            eval_output_dir = os.path.join(args.output_root, script_tag, dataset, ckpt_tag)
            resume_source_dir = os.path.join(args.finetune_output_root, script_tag, dataset, ckpt_tag)
            resume_ckpt = resolve_ckpt(resume_source_dir)
            if resume_ckpt is None and args.eval_pretrain:
                resume_ckpt = pretrain_ckpt

            progress_key = f"{dataset}::{script_tag}::{ckpt_tag}"
            if args.resume_progress and not args.force:
                prior = progress.get(progress_key)
                if prior and prior.get("status") == "ok":
                    summary_rows.append({
                        "dataset": dataset,
                        "model": script_tag,
                        "ckpt": ckpt_tag,
                        "output_dir": eval_output_dir,
                        "top1": prior.get("top1"),
                        "top5": prior.get("top5"),
                        "status": "skipped(ok)",
                    })
                    continue

            if resume_ckpt is None:
                print(f"[WARN] Skip {dataset} {script_tag}: missing checkpoint in {eval_output_dir}")
                record = {
                    "dataset": dataset,
                    "model": script_tag,
                    "ckpt": ckpt_tag,
                    "output_dir": eval_output_dir,
                    "top1": None,
                    "top5": None,
                    "status": "missing_ckpt",
                }
                summary_rows.append(record)
                progress[progress_key] = {**record, "resume_ckpt": resume_ckpt}
                save_progress(progress_file, progress)
                continue

            ret = run_eval(
                training_dir=args.training_dir,
                dataset=dataset,
                dataset_path=dataset_path,
                output_dir=eval_output_dir,
                resume_ckpt=resume_ckpt,
                pretrain_ckpt=pretrain_ckpt,
                model_name=model_name,
                nb_classes=nb_classes,
                args=args,
                extra_flag=extra_flag,
            )
            if ret != 0:
                record = {
                    "dataset": dataset,
                    "model": script_tag,
                    "ckpt": ckpt_tag,
                    "output_dir": eval_output_dir,
                    "top1": None,
                    "top5": None,
                    "status": f"failed({ret})",
                }
                summary_rows.append(record)
                progress[progress_key] = {**record, "resume_ckpt": resume_ckpt}
                save_progress(progress_file, progress)
                continue

            top1, top5 = parse_log_metrics(eval_output_dir)
            record = {
                "dataset": dataset,
                "model": script_tag,
                "ckpt": ckpt_tag,
                "output_dir": eval_output_dir,
                "top1": top1,
                "top5": top5,
                "status": "ok",
            }
            summary_rows.append(record)
            progress[progress_key] = {**record, "resume_ckpt": resume_ckpt}
            save_progress(progress_file, progress)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = output_root / f"summary_{timestamp}.json"
    summary_csv = output_root / f"summary_{timestamp}.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "ckpt", "output_dir", "top1", "top5", "status"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[INFO] Summary written: {summary_json}")
    print(f"[INFO] Summary written: {summary_csv}")
    print(f"[INFO] Progress written: {progress_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
