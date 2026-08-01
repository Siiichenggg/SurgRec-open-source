#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

FINETUNE_RE = re.compile(
    r"^\[INFO\]\s+Finetune:\s+(?P<model>.+?)\s+/\s+(?P<dataset>.+?)\s+\(task=(?P<task>[^,]+),\s*fold=(?P<fold>\d+)\)"
)
ERROR_RE = re.compile(r"^\[ERROR\]\s+Test failed:\s+(?P<model>.+?)\s+/\s+(?P<dataset>.+)$")
ACC_LINE_RE = re.compile(r"\*\s+Acc@1\s+(?P<acc1>[0-9.]+)\s+Acc@5\s+(?P<acc5>[0-9.]+)\s+loss\s+(?P<loss>[0-9.]+)")
ACCURACY_LINE_RE = re.compile(
    r"Accuracy of the network on the\s+(?P<count>\d+)\s+test videos:\s+Top-1:\s*(?P<top1>[0-9.]+)%,\s+Top-5:\s*(?P<top5>[0-9.]+)%"
)

@dataclass
class BlockResult:
    model: str
    dataset: str
    task: str
    fold: int
    acc1: Optional[float] = None
    acc5: Optional[float] = None
    loss: Optional[float] = None
    status: str = "ok"
    source: Optional[str] = None
    block_start_line: Optional[int] = None
    metric_line: Optional[int] = None


def _finalize_block(block: Optional[BlockResult], results: List[BlockResult]) -> None:
    if block is None:
        return
    results.append(block)


def parse_log(path: str) -> List[BlockResult]:
    results: List[BlockResult] = []
    current: Optional[BlockResult] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")

            finetune_match = FINETUNE_RE.match(line)
            if finetune_match:
                _finalize_block(current, results)
                current = BlockResult(
                    model=finetune_match.group("model").strip(),
                    dataset=finetune_match.group("dataset").strip(),
                    task=finetune_match.group("task").strip(),
                    fold=int(finetune_match.group("fold")),
                    block_start_line=line_num,
                )
                continue

            if current is None:
                continue

            error_match = ERROR_RE.match(line)
            if error_match:
                current.status = "error"
                continue

            acc_line_match = ACC_LINE_RE.search(line)
            if acc_line_match:
                current.acc1 = float(acc_line_match.group("acc1"))
                current.acc5 = float(acc_line_match.group("acc5"))
                current.loss = float(acc_line_match.group("loss"))
                current.source = "acc_line"
                current.metric_line = line_num
                continue

            accuracy_match = ACCURACY_LINE_RE.search(line)
            if accuracy_match:
                current.acc1 = float(accuracy_match.group("top1"))
                current.acc5 = float(accuracy_match.group("top5"))
                current.loss = current.loss
                current.source = "accuracy_line"
                current.metric_line = line_num
                continue

    _finalize_block(current, results)
    return results


def filter_by_fold(results: Iterable[BlockResult], folds: List[int], include_missing: bool) -> List[BlockResult]:
    filtered: List[BlockResult] = []
    fold_set = set(folds)
    for item in results:
        if item.fold not in fold_set:
            continue
        if not include_missing and item.acc1 is None and item.acc5 is None:
            continue
        filtered.append(item)
    return filtered


def write_csv(results: List[BlockResult], out_path: Optional[str]) -> None:
    fieldnames = [
        "model",
        "dataset",
        "task",
        "fold",
        "acc1",
        "acc5",
        "loss",
        "status",
        "source",
        "block_start_line",
        "metric_line",
    ]
    handle = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(asdict(row))
    if out_path:
        handle.close()


def write_json(results: List[BlockResult], out_path: Optional[str]) -> None:
    payload = [asdict(r) for r in results]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text + "\n")


def parse_folds(value: str) -> List[int]:
    folds: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    if not folds:
        raise ValueError("No valid folds provided.")
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-model results for a given fold.")
    parser.add_argument("--log", required=True, help="Path to allfolds_train_test.log")
    parser.add_argument("--fold", required=True, help="Fold id, e.g. 0 or 0,1")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--out", default=None, help="Output file path (default: stdout)")
    parser.add_argument("--include-missing", action="store_true", help="Include blocks without metrics")

    args = parser.parse_args()
    folds = parse_folds(args.fold)

    results = parse_log(args.log)
    filtered = filter_by_fold(results, folds, args.include_missing)

    if args.format == "csv":
        write_csv(filtered, args.out)
    else:
        write_json(filtered, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
