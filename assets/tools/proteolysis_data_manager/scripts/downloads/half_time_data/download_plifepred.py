#!/usr/bin/env python3
"""Invoke the library download_half_life command from its canonical script location."""
from pathlib import Path
import sys

# Download destination configuration
LIBRARY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = LIBRARY_ROOT.parents[2] / "data/proteolysis/half_time_datasets"

sys.path.insert(0, str(LIBRARY_ROOT / "src"))
from proteolysis_data_manager.cli import download_half_life as main

if __name__ == "__main__":
    raise SystemExit(main(["--data-root", str(DEFAULT_DATA_ROOT), "--source", "plifepred", *sys.argv[1:]]))
