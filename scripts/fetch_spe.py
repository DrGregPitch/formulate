#!/usr/bin/env python3
"""Download the solid polymer electrolyte (SPE) dataset for the real-data campaign.

    python scripts/fetch_spe.py

Pulls the two CSVs from CheMixHub (MIT-licensed) into a gitignored cache. No data
is committed to this repo. Cite: CheMixHub (github.com/chemcognition-lab/chemixhub);
the SPE conductivity data is itself curated from the published literature.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/chemcognition-lab/chemixhub/main/"
        "datasets/polymer-electrolyte/processed_data/")
FILES = ["processed_PolymerElectrolyteData.csv", "compounds.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data_cache", type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = args.outdir / fname
        print(f"Fetching {fname} ...")
        urllib.request.urlretrieve(BASE + fname, dest)  # noqa: S310 - trusted https
        print(f"  saved {dest} ({dest.stat().st_size / 1024:.0f} KB)")

    print("\nLicense: MIT (CheMixHub). SPE conductivity data curated from the literature.")
    print("\nRun the real-data campaign:\n  python scripts/run_spe.py --outdir results")


if __name__ == "__main__":
    main()
