# Datasets, Splits, and Licenses

This repository releases **split metadata and code only** — lists of video
filenames and integer class labels. It does **not** redistribute any video
frames or pixels. Every dataset remains under its own license; use each one
according to its original terms, and cite the original authors.

## Split Format

```text
splits/<dataset>/
  train.csv
  val.csv
  test.csv
  label_map.json        # optional
  README.md / LICENSE   # optional, from the source dataset
```

Each CSV row is `<video_path> <label_id>`. The committed CSVs use
`__DATA_ROOT__` instead of a machine-specific path; rewrite them before training:

```bash
python tools/data/relocate_splits.py \
  --input-root splits --output-root data/splits_local \
  --data-root /path/to/your/dataset_root
```

Arrange the videos under `--data-root` using the same relative paths the split
metadata expects.

## Source Dataset Licenses

Licenses verified against each dataset's authoritative source (dataset homepage,
Zenodo/Synapse/Figshare record, UCI, or the source repository). "Access" is how
the videos are obtained; it does not change the metadata license but is relevant
to how freely derived files may be shared.

| Dataset | License | Access | Metadata redistribution |
| --- | --- | --- | --- |
| cataract-1k | CC BY 4.0 | free Synapse account | OK |
| Colonoscopic | CC BY 4.0 | open (UCI) | OK |
| Hyper-Kvasir | CC BY 4.0 | open | OK |
| Kvasir-Capsule | CC BY 4.0 | open | OK |
| cataract-101 | CC BY-NC 4.0 | open | OK (non-commercial) |
| LapGyn4 | CC BY-NC 4.0 | open | OK (non-commercial) |
| AutoLaparo | CC BY-NC-SA 4.0 | request form | OK; CSV inherits BY-NC-SA |
| cat-21 (ITEC) | CC BY-NC-SA 4.0 | open | OK; CSV inherits BY-NC-SA |
| MultiBypass140 | CC BY-NC-SA 4.0 | open | OK; CSV inherits BY-NC-SA |
| SAR-RARP50 | CC BY-NC-SA 4.0 | open | OK; CSV inherits BY-NC-SA |
| SurgicalActions160 | CC BY-NC-SA 4.0 | open | OK; CSV inherits BY-NC-SA |
| Cholec80 | CC BY-NC-SA 4.0 | request form | Allowed under BY-NC-SA; access is form-gated |
| M2CAI16-Workflow | CC BY-NC-SA 4.0 | request form | Allowed under BY-NC-SA; access is form-gated |
| AIxSuture | CC BY-NC-ND 4.0 | open (Zenodo) | With provider permission (see below) |
| JIGSAWS | custom (IRB release) | registration | With provider permission (see below) |
| LDPolypVideo | none stated | open | With provider permission (see below) |

### ShareAlike obligation

The datasets marked "inherits BY-NC-SA" are licensed CC BY-NC-SA 4.0. A derived
split file is adapted material, so those specific CSVs are redistributed here
under **CC BY-NC-SA 4.0** (not the repository-wide CC BY-NC 4.0), with
attribution to the source dataset.

### Splits redistributed with provider permission

Three sources do not grant redistribution through an open license — AIxSuture is
CC BY-NC-**ND** 4.0 (NoDerivatives), JIGSAWS is an IRB-approved registration-gated
release, and LDPolypVideo ships with no license statement. The SurgRec authors
obtained permission from the respective dataset providers to redistribute the
derived split metadata (filenames + labels) for non-commercial research use.

That permission was granted to the SurgRec authors for this release; it is not a
sublicense and does not travel downstream under this repository's CC BY-NC 4.0.
If you intend to redistribute these three manifests yourself, contact the
original providers. Downstream users must in all cases obtain the underlying
videos from each dataset's official source under its own terms.

## Attribution

Please cite the original dataset papers alongside this work. Dataset homepages:

- AIxSuture — https://zenodo.org/records/7940583
- AutoLaparo — https://autolaparo.github.io
- cat-21 (Cataract-21, ITEC) — http://ftp.itec.aau.at/datasets/ovid/cat-21/
- cataract-101 — http://ftp.itec.aau.at/datasets/ovid/cat-101/
- cataract-1k — see paper PMC11014927 (Synapse record)
- Cholec80 / M2CAI16 — https://camma.unistra.fr/datasets/
- Colonoscopic — https://archive.ics.uci.edu/dataset/408
- Hyper-Kvasir — https://datasets.simula.no/hyper-kvasir/
- JIGSAWS — https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- Kvasir-Capsule — https://datasets.simula.no/kvasir-capsule/
- LapGyn4 — http://ftp.itec.aau.at/datasets/LapGyn4/
- LDPolypVideo — https://github.com/dashishi/LDPolypVideo-Benchmark
- MultiBypass140 — https://github.com/CAMMA-public/MultiBypass140
- SAR-RARP50 — https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529
- SurgicalActions160 — https://ftp.itec.aau.at/datasets/SurgicalActions160/

## Auxiliary splits

`splits/` also contains PitVis, which is not part of the 16-dataset paper
benchmark and has not been license-reviewed here. Check its terms before use.

## Privacy

Do not redistribute protected health information or private clinical videos.
`splits/` contains only file listings and labels, never pixels.
