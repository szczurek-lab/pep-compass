from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import checkFasta


def BINARY(fastas: Sequence[Sequence[str]], **kw: object) -> int | list[object]:
    """Binary.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: int | list[object]
    """
    if checkFasta.checkFasta(fastas) == False:
        print(
            'Error: for "BINARY" encoding, the input fasta sequences should be with equal length. \n\n'
        )
        return 0

    AA = "ARNDCQEGHILKMFPSTWYV"
    encodings = []
    header = ["#"]
    for i in range(1, len(fastas[0][1]) * 20 + 1):
        header.append("BINARY.F" + str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for aa in sequence:
            if aa == "-":
                code = code + [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
                continue
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        encodings.append(code)
    return encodings
