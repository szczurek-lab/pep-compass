from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import os
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculateDicorderContent(pos: int, endPos: int, disValue: Sequence[float]) -> float:
    """Calculate dicorder content.

    :param pos: Pos.
    :type pos: int
    :param endPos: Endpos.
    :type endPos: int
    :param disValue: Disvalue.
    :type disValue: Sequence[float]
    :return: Result of this operation.
    :rtype: float
    """
    newValues = disValue[pos:endPos]
    return [
        newValues.count("D") / (endPos - pos),
        newValues.count("O") / (endPos - pos),
    ]


def DisorderC(fastas: Sequence[Sequence[str]], **kw: object) -> int | list[object]:
    """Disorder c.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: int | list[object]
    """
    encodings = []
    header = ["#", "disorder-content", "order-content"]
    encodings.append(header)

    disDir = kw["path"]
    if disDir == None:
        print(
            'Error: please specify the directory of predicted protein disorder files by "--path"'
        )
        return 0

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        if os.path.exists(disDir + "/" + name + ".dis") == False:
            print(
                "Error: the predicted disorder information file (.dis) for protein "
                + name
                + " does not exist."
            )
            return 0

        with open(disDir + "/" + name + ".dis") as f:
            records = f.readlines()
        tag = 0
        for i in range(len(records)):
            if re.search("^-------", records[i]):
                tag = i
                break
        records = records[tag + 1 : -1]

        proteinSeq = ""
        disValue = []
        for line in records:
            array = line.rstrip().split() if line.rstrip() != "" else None
            key = array[3] if array[3] == "D" else "O"
            proteinSeq = proteinSeq + array[1]
            disValue.append(key)

        pos = proteinSeq.find(sequence)
        if pos == -1:
            print("Warning: could not find the peptide in proteins.\n\n")
        else:
            code = code + calculateDicorderContent(pos, pos + len(sequence), disValue)
        encodings.append(code)

    return encodings
