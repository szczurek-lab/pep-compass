#!/usr/bin/env python3
"""Invoke the original window_neighbourhood_datasets acquisition command."""
from pathlib import Path
import sys

LIBRARY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = LIBRARY_ROOT.parents[2] / "data/proteolysis/window_neighbourhood_datasets"
sys.path.insert(0, str(LIBRARY_ROOT / "src"))
from proteolysis_data_manager.cli import download_window_neighbourhood as main

if __name__ == "__main__":
    raise SystemExit(main(["--data-root", str(DEFAULT_DATA_ROOT), "--source", "merops", *sys.argv[1:]]))
