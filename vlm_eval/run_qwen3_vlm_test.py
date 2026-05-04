#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


GENERIC_PARTS = {
    "split_video",
    "split_videos",
    "split_clip",
    "videos",
    "video",
    "video_faststart",
    "train",
    "test",
    "val",
    "validation",
    "phase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL batch test on dataset_split")
    parser.add_argument("--model-path", default="checkpoints/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset-root", default="data/splits")
    parser.add_argument("--output-root", default="vlm_eval/output_qwen3_vl_test")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
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


def parse_split_line(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    path, label = parts[0], parts[1]
    path = path.strip().strip('"').strip("'")
    try:
        label_id = int(label)
    except ValueError:
        return None
    return path, label_id


def read_split_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_split_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def clean_name(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"^\d+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown"


def normalize_text(text: str) -> str:
    text = clean_name(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_class_name(video_path: str, dataset_name: str) -> str:
    stem = Path(video_path).stem

    if dataset_name == "LDPolyVideo":
        stem_lower = stem.lower()
        if "no_polyps" in stem_lower or "nopolyps" in stem_lower or "no-polyps" in stem_lower:
            return "no polyps"
        if "polyps" in stem_lower or "polyp" in stem_lower:
            return "polyps"

    for marker in ("_phase_", "_step_"):
        if marker in stem:
            tail = stem.split(marker, 1)[1]
            parts = [p for p in tail.split("_") if p]
            if len(parts) >= 2 and parts[0].isdigit():
                label_tokens = []
                for tok in parts[1:]:
                    if re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", tok):
                        break
                    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
                        break
                    label_tokens.append(tok)
                if label_tokens:
                    return clean_name("_".join(label_tokens))

    parts = [p for p in Path(video_path).parts if p not in ("", os.sep)]
    dataset_token = dataset_name
    if dataset_token in parts:
        idx = parts.index(dataset_token)
        rel = parts[idx + 1 :]
    else:
        rel = parts

    if dataset_name in {"Colonoscopic-web", "cat-21", "cataract-101", "kvasir-capsule"}:
        if len(rel) >= 2:
            return clean_name(rel[-2])
    if dataset_name == "hyper-kvasir":
        if len(rel) >= 2:
            return clean_name(rel[-2])
    if dataset_name == "LapGyn_dataset":
        if len(rel) >= 1:
            return clean_name(rel[0])

    for token in reversed(rel[:-1]):
        token_lower = token.lower()
        if token_lower in GENERIC_PARTS:
            continue
        return clean_name(token)

    return clean_name(stem)


def _apply_dataset_label_overrides(dataset_name: str, name: str) -> str:
    normalized = normalize_text(name)

    if dataset_name == "M2CAI16-Workflow" and normalized == "phase":
        return "unspecified phase"

    return clean_name(name)


def _invert_label_mapping(name_to_id: dict, allowed_labels: set[int]):
    id_to_name = {}
    for name, label_id in name_to_id.items():
        try:
            lid = int(label_id)
        except Exception:
            continue
        if lid not in allowed_labels:
            continue
        if lid not in id_to_name:
            id_to_name[lid] = clean_name(str(name))
    return id_to_name


def _load_label_map_from_json(dataset_dir: Path, allowed_labels: set[int]):
    candidates = [dataset_dir / "label_map.json", dataset_dir / "label_mapping.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if isinstance(payload, dict):
            if "label_to_index" in payload and isinstance(payload.get("label_to_index"), dict):
                mapped = _invert_label_mapping(payload["label_to_index"], allowed_labels)
                if mapped:
                    return mapped, f"{path.name}:label_to_index"
            if "index_to_label" in payload and isinstance(payload.get("index_to_label"), dict):
                id_to_name = {}
                for k, v in payload["index_to_label"].items():
                    try:
                        lid = int(k)
                    except Exception:
                        continue
                    if lid in allowed_labels:
                        id_to_name[lid] = clean_name(str(v))
                if id_to_name:
                    return id_to_name, f"{path.name}:index_to_label"
            mapped = _invert_label_mapping(payload, allowed_labels)
            if mapped:
                return mapped, path.name
    return None, ""


def _literal_eval_assignment_dict(script_path: Path, var_name: str):
    try:
        code = script_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(code)
    except Exception:
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == var_name:
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    return None
                if isinstance(value, dict):
                    return value
    return None


def _load_label_map_from_split_scripts(dataset_dir: Path, allowed_labels: set[int]):
    scripts = sorted(dataset_dir.glob("split*.py"))
    for script in scripts:
        for var_name in ("PHASE_NAME_MAP", "LABEL_MAP"):
            mapping = _literal_eval_assignment_dict(script, var_name)
            if not isinstance(mapping, dict):
                continue
            id_to_name = _invert_label_mapping(mapping, allowed_labels)
            if id_to_name:
                return id_to_name, f"{script.name}:{var_name}"
    return None, ""


def _load_dataset_specific_label_map(dataset_name: str, allowed_labels: set[int]):
    dataset_static = {
        "AutoLaparo": {
            0: "Preparation",
            1: "Dividing Ligament and Peritoneum",
            2: "Dividing Uterine Vessels and Ligament",
            3: "Transecting the Vagina",
            4: "Specimen Removal",
            5: "Suturing",
            6: "Washing",
        },
        "JIGSAWS": {
            0: "Reach for needle (G1)",
            1: "Position needle (G2)",
            2: "Push needle through tissue (G3)",
            3: "Transfer needle between hands (G4)",
            4: "Move to center with needle in grip (G5)",
            5: "Pull suture with left hand (G6)",
            6: "Orient needle (G8)",
            7: "Use right hand to help tighten suture (G9)",
            8: "Loosen more suture (G11)",
            9: "Drop suture at end and move to endpoints (G12)",
            10: "Hold needle with left hand (G13)",
            11: "Pull suture with right hand (G14)",
            12: "Orient needle with right hand (G15)",
        },
    }

    mapping = dataset_static.get(dataset_name)
    if not mapping:
        return None, ""

    filtered = {label: clean_name(name) for label, name in mapping.items() if label in allowed_labels}
    if not filtered:
        return None, ""
    return filtered, "dataset_specific_mapping"


def build_label_map(train_rows, dataset_name: str, dataset_dir: Path):
    counter = defaultdict(Counter)
    labels = sorted({label for _, label in train_rows})

    allowed = set(labels)
    json_map, json_source = _load_label_map_from_json(dataset_dir, allowed)
    if json_map is not None:
        label_map = {}
        for label in labels:
            raw_name = json_map.get(label, f"class {label}")
            label_map[label] = _apply_dataset_label_overrides(dataset_name, raw_name)
        return label_map, json_source

    script_map, script_source = _load_label_map_from_split_scripts(dataset_dir, allowed)
    if script_map is not None:
        label_map = {}
        for label in labels:
            raw_name = script_map.get(label, f"class {label}")
            label_map[label] = _apply_dataset_label_overrides(dataset_name, raw_name)
        return label_map, script_source

    specific_map, specific_source = _load_dataset_specific_label_map(dataset_name, allowed)
    if specific_map is not None:
        label_map = {}
        for label in labels:
            raw_name = specific_map.get(label, f"class {label}")
            label_map[label] = _apply_dataset_label_overrides(dataset_name, raw_name)
        return label_map, specific_source

    for path, label in train_rows:
        cname = infer_class_name(path, dataset_name)
        counter[label][cname] += 1

    label_map = {}
    for label in labels:
        if counter[label]:
            raw_name = counter[label].most_common(1)[0][0]
            label_map[label] = _apply_dataset_label_overrides(dataset_name, raw_name)
        else:
            label_map[label] = _apply_dataset_label_overrides(dataset_name, f"class_{label}")
    return label_map, "path_inference"


def load_middle_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        mid = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def get_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def make_prompt(label_map, variant: str = "baseline"):
    if variant == "stability_v1":
        lines = ["You are an expert medical visual classifier for surgical/endoscopy frames."]
        lines.append("You will receive one representative frame from a short procedure clip.")
        lines.append("Pick exactly one best-matching class from the options.")
        lines.append("Output strictly one token: either the option letter (A/B/C/...) or the exact class name.")
    else:
        lines = ["You are a medical video classifier."]
        lines.append("Given one representative frame from a short surgical/endoscopy clip, choose exactly one option.")
        lines.append("Return ONLY the option letter (A/B/C/...) or the exact class name.")
    lines.append("Multiple-choice options:")
    for i, label_id in enumerate(sorted(label_map.keys())):
        option = chr(ord("A") + i)
        lines.append(f"{option}. {label_map[label_id]}")
    return "\n".join(lines)


def extract_pred_label(text: str, label_map):
    text = (text or "").strip()

    option_to_label = {}
    for i, label_id in enumerate(sorted(label_map.keys())):
        option_to_label[chr(ord("A") + i)] = label_id

    letter_match = re.search(r"\b([A-Z])\b", text.upper())
    if letter_match and letter_match.group(1) in option_to_label:
        return option_to_label[letter_match.group(1)]

    normalized_output = normalize_text(text)
    name_candidates = []
    for label_id, name in label_map.items():
        normalized_name = normalize_text(name)
        if normalized_name and normalized_name in normalized_output:
            name_candidates.append((len(normalized_name), label_id))
    if name_candidates:
        name_candidates.sort(reverse=True)
        return name_candidates[0][1]

    m = re.search(r"-?\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    if args.start_dataset:
        if args.start_dataset in all_datasets:
            start_idx = all_datasets.index(args.start_dataset)
            all_datasets = all_datasets[start_idx:]

    if args.num_workers > 1:
        all_datasets = all_datasets[args.worker_id :: args.num_workers]

    if not all_datasets:
        print(
            f"[INFO] worker {args.worker_id}/{args.num_workers}: no datasets assigned, exiting.",
            flush=True,
        )
        return

    print(
        f"[INFO] worker {args.worker_id}/{args.num_workers} on {target_device}: "
        f"{len(all_datasets)} datasets assigned",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=get_dtype(args.dtype),
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
                    existing = json.load(f)
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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
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


if __name__ == "__main__":
    main()
