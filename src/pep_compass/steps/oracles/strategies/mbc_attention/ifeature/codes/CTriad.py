from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import re

"""
import sys, platform

def CTriad(fastas, **kw):
    dataFile = sys.path[0] + r'\data\CTD.txt' if platform.system() == 'Windows' else sys.path[0] + '/data/CTD.txt'
    CTDIndex = {}
    with open(dataFile) as f:
        records = f.readlines()
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        CTDIndex[array[0]] = int(array[1])

    encodings = []
    header = ['#']
    for i in range(1, 344):
        header.append('CTriad.' + str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name] + [0] * 343
        myDict = {}
        for j in range(len(sequence) - 2):
            key = sequence[j:j+3]
            myDict[key] = myDict[key] + 1 if key in myDict else 1
        for key in myDict:
            code[CTDIndex[key]] = code[CTDIndex[key]] + myDict[key]
        maxValue, minValue = max(code[1:]), min(code[1:])
        for j in range(1, 344):
            code[j] = (code[j] - minValue) / maxValue
        encodings.append(code)
    return encodings
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch


def CalculateKSCTriad(
    sequence: str, gap: int, features: torch.Tensor, AADict: Mapping[str, int]
) -> list[object]:
    """Calculate ksctriad.

    :param sequence: One peptide sequence.
    :type sequence: str
    :param gap: Gap.
    :type gap: int
    :param features: Features.
    :type features: torch.Tensor
    :param AADict: Aadict.
    :type AADict: Mapping[str, int]
    :return: Result of this operation.
    :rtype: list[object]
    """
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + gap + 1 < len(sequence) and i + 2 * gap + 2 < len(sequence):
                fea = (
                    AADict[sequence[i]]
                    + "."
                    + AADict[sequence[i + gap + 1]]
                    + "."
                    + AADict[sequence[i + 2 * gap + 2]]
                )
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def CTriad(
    fastas: Sequence[Sequence[str]], gap: int = 0, **kw: object
) -> int | list[object]:
    """Ctriad.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param gap: Gap.
    :type gap: int
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: int | list[object]
    """
    AAGroup = {
        "g1": "AGV",
        "g2": "ILFP",
        "g3": "YMTS",
        "g4": "HNQW",
        "g5": "RK",
        "g6": "DE",
        "g7": "C",
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [
        f1 + "." + f2 + "." + f3
        for f1 in myGroups
        for f2 in myGroups
        for f3 in myGroups
    ]

    encodings = []
    header = ["#"]
    for f in features:
        header.append(f)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        if len(sequence) < 3:
            print(
                'Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n'
            )
            return 0
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return encodings
