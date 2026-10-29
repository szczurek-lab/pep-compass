#!/usr/bin/env python3
"""Invoke the library normalize_kinetics command from its canonical script location."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from proteolysis_data_manager.cli import normalize_kinetics as main

if __name__ == "__main__":
    raise SystemExit(main())
