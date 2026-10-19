from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch


#!/usr/bin/env python
# _*_coding:utf-8_*_

USAGE = """
USAGE:
	python KSCTriad.py input.fasta <K> output
	
	input.fasta:  the input protein sequence file in fasta format.
	K:            the max space number, integer, defaule: 5
	output:       the encoding file, default: 'encodings.tsv'
"""
import os
import re
import sys

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import readFasta
import saveCode


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
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = (
                    AADict[sequence[i]]
                    + "."
                    + AADict[sequence[i + g + 1]]
                    + "."
                    + AADict[sequence[i + 2 * g + 2]]
                )
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def KSCTriad(
    fastas: Sequence[Sequence[str]], gap: int = 0, **kw: object
) -> int | list[object]:
    """Ksctriad.

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
    for g in range(gap + 1):
        for f in features:
            header.append(f + ".gap" + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        if len(sequence) < 2 * gap + 3:
            print(
                'Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n'
            )
            return 0
        code = code + CalculateKSCTriad(sequence, gap, features, AADict)
        encodings.append(code)

    return encodings


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(USAGE)
        sys.exit(1)
    fastas = readFasta.readFasta(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    output = sys.argv[3] if len(sys.argv) >= 4 else "encoding.tsv"
    encodings = KSCTriad(fastas, k)
    saveCode.savetsv(encodings, output)
