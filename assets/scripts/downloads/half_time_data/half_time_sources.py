"""Compatibility import for existing serum downloader callers."""
import _bootstrap
import sys
import importlib
sys.modules[__name__] = importlib.import_module("proteolysis_data_manager.acquisition.half_life")
