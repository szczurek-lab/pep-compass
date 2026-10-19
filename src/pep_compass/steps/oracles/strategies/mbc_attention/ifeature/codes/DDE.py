from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def DDE(fastas: Sequence[Sequence[str]], **kw: object) -> list[object]:
    """Dde.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: list[object]
    """
    AA = kw["order"] if kw["order"] != None else "ACDEFGHIKLMNPQRSTVWY"

    myCodons = {
        "A": 4,
        "C": 2,
        "D": 2,
        "E": 2,
        "F": 2,
        "G": 4,
        "H": 2,
        "I": 3,
        "K": 2,
        "L": 6,
        "M": 1,
        "N": 2,
        "P": 4,
        "Q": 2,
        "R": 6,
        "S": 6,
        "T": 4,
        "V": 4,
        "W": 1,
        "Y": 2,
    }

    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ["#"] + diPeptides
    encodings.append(header)

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = (
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] + 1
            )
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)
    return encodings
