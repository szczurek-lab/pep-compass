"""Allow checkout entry scripts to import the independent assets/tools package."""
from pathlib import Path
import sys

PACKAGE_SRC = Path(__file__).resolve().parents[4] / 'assets/tools/proteolysis_data_manager/src'
sys.path.insert(0, str(PACKAGE_SRC))
