from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def readCode(file: str | Path) -> np.ndarray:
    """Read code.

    :param file: File.
    :type file: str | Path
    :return: Result of this operation.
    :rtype: np.ndarray
    """
    encodings = []
    if not os.path.exists(file):
        print("Error: file does not exist.")
        sys.exit(1)
    with open(file) as f:
        records = f.readlines()
    for i in records:
        array = i.rstrip().split() if i.strip() != "" else None
        encodings.append(array)
    return np.array(encodings)
