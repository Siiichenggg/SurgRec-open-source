"""relocate_splits rewrites the __DATA_ROOT__ placeholder in the shipped CSVs.

Every downstream run reads the rewritten files, so a silent failure here points
training at paths that do not exist.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_relocate(tmp_path, data_root="/data/surgical"):
    out = tmp_path / "splits_local"
    subprocess.run(
        [sys.executable, str(REPO / "tools/data/relocate_splits.py"),
         "--input-root", str(REPO / "splits"),
         "--output-root", str(out),
         "--data-root", data_root],
        check=True, capture_output=True,
    )
    return out


def test_placeholder_is_replaced(tmp_path):
    out = run_relocate(tmp_path)
    csvs = list(out.rglob("*.csv"))
    assert csvs, "no split CSVs were written"
    for csv in csvs:
        text = csv.read_text()
        assert "__DATA_ROOT__" not in text, f"{csv} still holds the placeholder"


def test_paths_become_absolute_under_data_root(tmp_path):
    out = run_relocate(tmp_path, "/mnt/videos")
    sample = out / "cholec80" / "train.csv"
    first = sample.read_text().splitlines()[0]
    path, label = first.rsplit(" ", 1)
    assert path.startswith("/mnt/videos/")
    assert label.isdigit()


def test_every_shipped_split_is_carried_over(tmp_path):
    out = run_relocate(tmp_path)
    shipped = {p.relative_to(REPO / "splits") for p in (REPO / "splits").rglob("*.csv")}
    written = {p.relative_to(out) for p in out.rglob("*.csv")}
    assert shipped == written
