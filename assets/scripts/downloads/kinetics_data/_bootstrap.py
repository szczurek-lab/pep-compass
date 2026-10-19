"""Make the independent data-manager package available from a checkout."""
import sys
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[4] / 'assets/tools/proteolysis_data_manager/src'
sys.path.insert(0, str(PACKAGE_SRC))
