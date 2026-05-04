#!/usr/bin/env python3
"""Relocate sanitized SurgRec split CSVs to a local dataset root."""
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='splits', help='Root containing placeholder split CSVs')
    parser.add_argument('--output-root', required=True, help='Directory for rewritten CSVs')
    parser.add_argument('--data-root', required=True, help='Absolute root replacing __DATA_ROOT__')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    data_root = str(Path(args.data_root).expanduser().resolve())

    if not input_root.is_dir():
        raise SystemExit(f'Input root not found: {input_root}')

    for src in input_root.rglob('*'):
        if not src.is_file():
            continue
        rel = src.relative_to(input_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding='utf-8', errors='replace')
        if src.suffix == '.csv':
            text = text.replace('__DATA_ROOT__', data_root)
        dst.write_text(text, encoding='utf-8')
    print(f'Wrote relocated splits to {output_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
