#!/usr/bin/env python3
"""Acquire raw data using the repository's configured series directories."""
from pathlib import Path
import sys

# Download destination configuration
LIBRARY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = LIBRARY_ROOT.parents[2]
DEFAULT_HALF_LIFE_ROOT = REPOSITORY_ROOT / "data/proteolysis/half_time_datasets"
DEFAULT_WINDOW_ROOT = REPOSITORY_ROOT / "data/proteolysis/window_neighbourhood_datasets"
DEFAULT_KINETICS_ROOT = REPOSITORY_ROOT / "data/proteolysis/kinetics_datasets"

sys.path.insert(0, str(LIBRARY_ROOT / "src"))
from proteolysis_data_manager.cli import download_raw as main

if __name__ == "__main__":
    raise SystemExit(main([
        "--half-life-root", str(DEFAULT_HALF_LIFE_ROOT),
        "--kinetics-root", str(DEFAULT_KINETICS_ROOT),
        "--window-root", str(DEFAULT_WINDOW_ROOT),
        *sys.argv[1:],
    ]))
