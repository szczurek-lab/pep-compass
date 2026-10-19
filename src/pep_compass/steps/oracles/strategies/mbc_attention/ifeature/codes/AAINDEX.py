from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import os
import platform
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import checkFasta


def AAINDEX(fastas: Sequence[Sequence[str]], **kw: object) -> int | list[object]:
    """Aaindex.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: int | list[object]
    """
    if checkFasta.checkFasta(fastas) == False:
        print(
            'Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n'
        )
        return 0

    AA = "ARNDCQEGHILKMFPSTWYV"

    fileAAindex = (
        re.sub("codes$", "", os.path.split(os.path.realpath(__file__))[0])
        + r"\data\AAindex.txt"
        if platform.system() == "Windows"
        else re.sub("codes$", "", os.path.split(os.path.realpath(__file__))[0])
        + "/data/AAindex.txt"
    )
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != "" else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != "" else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    encodings = []
    header = ["#"]
    for pos in range(1, len(fastas[0][1]) + 1):
        for idName in AAindexName:
            header.append("SeqPos." + str(pos) + "." + idName)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for aa in sequence:
            if aa == "-":
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encodings.append(code)

    return encodings
