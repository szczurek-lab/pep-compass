#!/usr/bin/env python3
"""Invoke the library's common dataset build without duplicating conversion logic."""

import logging
import sys
from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(LIBRARY_ROOT / "src"))
from proteolysis_data_manager.integration.merge.build import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    raise SystemExit(main())
