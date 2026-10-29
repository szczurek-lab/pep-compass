#!/usr/bin/env python3
"""Build the peptide and experiment datasets through the library command."""
from pathlib import Path
import sys

# Dataset destination configuration
LIBRARY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = LIBRARY_ROOT.parents[2] / "data/proteolysis/half_time_datasets"

sys.path.insert(0, str(LIBRARY_ROOT / "src"))
from proteolysis_data_manager.cli import normalize_half_life as main

if __name__ == "__main__":
    raise SystemExit(main(["--data-root", str(DEFAULT_DATA_ROOT), *sys.argv[1:]]))
