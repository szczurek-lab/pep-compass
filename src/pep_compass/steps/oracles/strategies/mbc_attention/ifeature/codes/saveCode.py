from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def savetsv(
    encodings: Sequence[Sequence[str | float | int]], file: str = "encoding.tsv"
) -> None:
    """Savetsv.

    :param encodings: Feature table including the identifiers and numeric columns expected by this operation.
    :type encodings: Sequence[Sequence[str | float | int]]
    :param file: File.
    :type file: str
    :rtype: None
    """
    with open(file, "w") as f:
        if encodings == 0:
            f.write("Descriptor calculation failed.")
        else:
            for i in range(len(encodings[0]) - 1):
                f.write(encodings[0][i] + "\t")
            f.write(encodings[0][-1] + "\n")
            for i in encodings[1:]:
                f.write(i[0] + "\t")
                for j in range(1, len(i) - 1):
                    f.write(str(float(i[j])) + "\t")
                f.write(str(float(i[len(i) - 1])) + "\n")
    return None
