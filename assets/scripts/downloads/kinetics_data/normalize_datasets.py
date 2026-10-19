#!/usr/bin/env python3
"""Thin entry point for the common relational kinetics contract."""
import _bootstrap
from proteolysis_data_manager.cli import normalize_kinetics as main

if __name__ == '__main__':
    raise SystemExit(main())
