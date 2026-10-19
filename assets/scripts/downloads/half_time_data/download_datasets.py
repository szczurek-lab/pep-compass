#!/usr/bin/env python3
"""Checkout entry point for the independent proteolysis data manager."""
import _bootstrap  # Register the assets/tools source checkout.
from proteolysis_data_manager.cli import download_half_life as main


if __name__ == "__main__":
    raise SystemExit(main())
