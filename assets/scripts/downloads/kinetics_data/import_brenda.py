#!/usr/bin/env python3
"""Thin entry point; acquisition lives in proteolysis_data_manager."""
import _bootstrap
from proteolysis_data_manager.cli import import_brenda as main

if __name__ == '__main__':
    raise SystemExit(main())
