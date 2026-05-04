#!/usr/bin/env python3
"""Check video readability with decord and optionally fix via ffmpeg.

Usage examples:
  python check_and_fix_videos.py --csv /path/to/test.csv --check
  python check_and_fix_videos.py --csv /path/to/test.csv --fix --strategy both

CSV format: each line starts with absolute video path, followed by label.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from decord import VideoReader, cpu  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    print(f"Failed to import decord: {exc}", file=sys.stderr)
    sys.exit(2)

DEFAULT_EOF_RETRY_MAX = "20480"


def read_csv_paths(csv_path: Path) -> List[Path]:
    paths: List[Path] = []
    with csv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = line.split()[0]
            paths.append(Path(path))
    return paths


def check_decord_readable(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"
    if path.stat().st_size < 1024:
        return False, "tiny_file"

    os.environ.setdefault("DECORD_EOF_RETRY_MAX", DEFAULT_EOF_RETRY_MAX)
    try:
        vr = VideoReader(str(path), num_threads=1, ctx=cpu(0))
        length = len(vr)
        if length <= 0:
            return False, "empty_video"
        try:
            _ = vr.get_batch([0]).asnumpy()
        except Exception as exc:
            return False, f"decord_head_error: {type(exc).__name__}"
        try:
            _ = vr.get_batch([max(0, length - 1)]).asnumpy()
        except Exception as exc:
            return False, f"decord_tail_error: {type(exc).__name__}"
        return True, "ok"
    except Exception as exc:
        return False, f"decord_open_error: {type(exc).__name__}"


def run_ffmpeg(cmd: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr.strip().splitlines()[-1] if proc.stderr else "ffmpeg_failed"
    except Exception as exc:
        return False, str(exc)


def remux_video(path: Path) -> Tuple[bool, str]:
    tmp_path = path.with_suffix(path.suffix + ".remux.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c",
        "copy",
        "-map",
        "0",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    ok, msg = run_ffmpeg(cmd)
    if ok:
        tmp_path.replace(path)
    else:
        if tmp_path.exists():
            tmp_path.unlink()
    return ok, msg


def reencode_video(path: Path) -> Tuple[bool, str]:
    tmp_path = path.with_suffix(path.suffix + ".reencode.mp4")
    cmd_candidates = [
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-c:v",
            "libopenh264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ],
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ],
    ]

    last_msg = ""
    for cmd in cmd_candidates:
        ok, msg = run_ffmpeg(cmd)
        if ok:
            tmp_path.replace(path)
            return True, "reencode_ok"
        last_msg = msg
        if tmp_path.exists():
            tmp_path.unlink()

    return False, last_msg


def fix_video(path: Path, strategy: str) -> Tuple[bool, str]:
    if strategy == "remux":
        return remux_video(path)
    if strategy == "reencode":
        return reencode_video(path)

    ok, msg = remux_video(path)
    if ok:
        return True, "remux_ok"
    ok, msg2 = reencode_video(path)
    if ok:
        return True, "reencode_ok"
    return False, msg or msg2


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and fix videos for decord.")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with video paths and labels")
    parser.add_argument("--check", action="store_true", help="Only check and report problems")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix problematic videos")
    parser.add_argument(
        "--strategy",
        choices=["remux", "reencode", "both"],
        default="both",
        help="Fix strategy when --fix is set",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to process")
    parser.add_argument("--report", type=str, default="", help="Write report to file")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    if not args.check and not args.fix:
        print("Nothing to do: specify --check and/or --fix", file=sys.stderr)
        return 1

    paths = read_csv_paths(csv_path)
    if args.limit > 0:
        paths = paths[: args.limit]

    problems: List[Tuple[Path, str]] = []
    fixed: List[Tuple[Path, str]] = []

    for idx, path in enumerate(paths, 1):
        ok, reason = check_decord_readable(path)
        if ok:
            continue
        problems.append((path, reason))
        if args.fix:
            fixed_ok, fix_msg = fix_video(path, args.strategy)
            if fixed_ok:
                fixed.append((path, fix_msg))
            else:
                fixed.append((path, f"fix_failed: {fix_msg}"))

    print(f"checked: {len(paths)}")
    print(f"problems: {len(problems)}")
    if args.fix:
        print(f"fixed_attempts: {len(fixed)}")
        print(f"fixed_success: {sum(1 for _, msg in fixed if msg in ('remux_ok', 'reencode_ok'))}")

    lines: List[str] = []
    for path, reason in problems:
        lines.append(f"{path}\t{reason}")
    if args.fix:
        for path, msg in fixed:
            lines.append(f"FIX\t{path}\t{msg}")

    if args.report:
        report_path = Path(args.report)
        report_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    else:
        if lines:
            print("\n".join(lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
