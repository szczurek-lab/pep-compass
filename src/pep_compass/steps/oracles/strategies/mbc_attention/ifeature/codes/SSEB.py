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


def SSEB(fastas: Sequence[Sequence[str]], **kw: object) -> int | list[object]:
    """Sseb.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: int | list[object]
    """
    if checkFasta.checkFasta(fastas) == False:
        print(
            'Error: for "SSEB" encoding, the input fasta sequences should be with equal length. \n\n'
        )
        return 0

    ssDir = kw["path"]
    if ssDir == None:
        print(
            'Error: please specify the directory of predicted protein disorder files by "--path"'
        )
        return 0

    encodings = []
    header = ["#"]
    for p in range(1, len(fastas[0][1]) + 1):
        for ss in ("H", "E", "C"):
            header.append("Pos" + str(p) + "." + ss)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        if os.path.exists(ssDir + "/" + name + ".ss2") == True:
            with open(ssDir + "/" + name + ".ss2") as f:
                records = f.readlines()[2:]
        elif os.path.exists(ssDir + "/" + name + ".spXout") == True:
            with open(ssDir + "/" + name + ".spXout") as f:
                records = f.readlines()[1:]
        else:
            print(
                "Error: the predicted secondary structure (.ss2 or .spXout) for protein "
                + name
                + " does not exist."
            )
            return 0

        proteinSeq = ""
        SSE = []
        myDict = {"H": [0, 0, 1], "E": [0, 1, 0], "C": [1, 0, 0]}
        for line in records:
            array = line.strip().split() if line.rstrip() != "" else None
            proteinSeq = proteinSeq + array[1]
            SSE.append(array[2])

        pos = proteinSeq.find(sequence)
        if pos == -1:
            print("Warning: could not find the peptide in proteins.\n\n")
        else:
            for p in range(pos, pos + len(sequence)):
                code = code + myDict[SSE[p]]
        encodings.append(code)

    return encodings
