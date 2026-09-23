#!/usr/bin/env python3
"""Reject ad-hoc SABIO-RK queries outside the fixed serum-protease manifest."""

from __future__ import annotations

import sys


def main() -> int:
    """Explain the supported serum-calibration SABIO-RK entry point.

    :return: Non-zero status because ad-hoc queries violate the acquisition contract.
    """

    print(
        "error: ad-hoc SABIO-RK queries are disabled for serum calibration. "
        "Build and map the fixed serum manifest, then run download_sabio_serum_panel.py.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
