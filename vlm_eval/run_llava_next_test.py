#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from run_qwen3_vlm_test import (
    build_label_map,
    extract_pred_label,
    get_dtype,
    load_middle_frame,
    make_prompt,
    read_split_csv,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA-NeXT batch test on dataset_split")
    parser.add_argument("--model-path", default="checkpoints/LLaVA-NeXT-vicuna-7b")
    parser.add_argument("--dataset-root", default="data/splits")
    parser.add_argument("--output-root", default="vlm_eval/output_llava_next_test")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples-per-dataset", type=int, default=0)
    parser.add_argument("--start-dataset", default="")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--prompt-variant",
        default="baseline",
        choices=["baseline", "stability_v1"],
        help="Prompt template variant for sensitivity testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("--worker-id must satisfy 0 <= worker-id < num-workers")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    try:
        cuda_index = int(args.cuda_device)
    except ValueError as exc:
        raise ValueError("--cuda-device must be an integer GPU index") from exc

    if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
        raise ValueError(
            f"--cuda-device={cuda_index} is invalid; available GPU count is {torch.cuda.device_count()}"
        )

    target_device = f"cuda:{cuda_index}"

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_datasets = []
    for d in sorted(dataset_root.iterdir()):
        if d.is_dir() and (d / "train.csv").is_file() and (d / "test.csv").is_file():
            all_datasets.append(d.name)

    if args.start_dataset and args.start_dataset in all_datasets:
        start_idx = all_datasets.index(args.start_dataset)
        all_datasets = all_datasets[start_idx:]

    if args.num_workers > 1:
        all_datasets = all_datasets[args.worker_id :: args.num_workers]

    if not all_datasets:
        print(f"[INFO] worker {args.worker_id}/{args.num_workers}: no datasets assigned, exiting.", flush=True)
        return

    print(
        f"[INFO] worker {args.worker_id}/{args.num_workers} on {target_device}: "
        f"{len(all_datasets)} datasets assigned",
        flush=True,
    )

    processor = LlavaNextProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=get_dtype(args.dtype),
        device_map=target_device,
    )
    model.eval()

    summary = []
    for dataset_name in all_datasets:
        dpath = dataset_root / dataset_name
        train_rows = read_split_csv(dpath / "train.csv")
        test_rows = read_split_csv(dpath / "test.csv")

        if args.max_samples_per_dataset and args.max_samples_per_dataset > 0:
            test_rows = test_rows[: args.max_samples_per_dataset]

        label_map, label_source = build_label_map(train_rows, dataset_name, dpath)
        prompt = make_prompt(label_map, args.prompt_variant)

        out_dir = output_root / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_json = out_dir / "predictions.json"
        meta_json = out_dir / "meta.json"

        existing = []
        done_keys = set()
        if args.resume and pred_json.is_file():
            try:
                with pred_json.open("r", encoding="utf-8") as f:
                    existing = csv_or_json_load(f)
                for item in existing:
                    done_keys.add(item.get("video_path", ""))
            except Exception:
                existing = []
                done_keys = set()

        predictions = list(existing)
        processed = 0
        for idx, (video_path, gt_label) in enumerate(test_rows):
            if video_path in done_keys:
                continue

            item = {
                "index": idx,
                "video_path": video_path,
                "gt_label": gt_label,
                "pred_label": None,
                "raw_output": "",
                "ok": False,
                "error": "",
            }

            if not os.path.isfile(video_path):
                item["error"] = "missing_video"
                predictions.append(item)
                processed += 1
                if processed % args.save_every == 0:
                    save_json(pred_json, predictions)
                continue

            image = load_middle_frame(video_path)
            if image is None:
                item["error"] = "decode_failed"
                predictions.append(item)
                processed += 1
                if processed % args.save_every == 0:
                    save_json(pred_json, predictions)
                continue

            text_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = processor(text=text_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            pred_label = extract_pred_label(output_text, label_map)

            item["pred_label"] = pred_label
            item["raw_output"] = output_text
            item["ok"] = pred_label == gt_label
            if pred_label is None:
                item["error"] = "parse_failed"

            predictions.append(item)
            processed += 1
            if processed % args.save_every == 0:
                save_json(pred_json, predictions)

        save_json(pred_json, predictions)

        valid = [p for p in predictions if p.get("pred_label") is not None]
        total = len(predictions)
        correct = sum(1 for p in predictions if p.get("ok"))
        valid_count = len(valid)
        accuracy = (correct / total) if total else 0.0
        valid_accuracy = (sum(1 for p in valid if p.get("ok")) / valid_count) if valid_count else 0.0

        meta = {
            "dataset": dataset_name,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "valid_count": valid_count,
            "valid_accuracy": valid_accuracy,
            "label_map": {str(k): v for k, v in sorted(label_map.items())},
            "label_source": label_source,
            "prompt_variant": args.prompt_variant,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(meta_json, meta)
        summary.append(meta)

    suffix = f"worker{args.worker_id}_of_{args.num_workers}"
    summary_json = output_root / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.json"
    save_json(summary_json, summary)

    summary_csv = output_root / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "total", "correct", "accuracy", "valid_count", "valid_accuracy", "timestamp"],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    print(f"[INFO] Done. Summary JSON: {summary_json}")
    print(f"[INFO] Done. Summary CSV: {summary_csv}")


def csv_or_json_load(file_obj):
    import json

    return json.load(file_obj)


if __name__ == "__main__":
    main()
