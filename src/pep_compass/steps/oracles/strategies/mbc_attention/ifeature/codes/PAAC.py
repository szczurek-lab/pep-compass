from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import math
import os
import platform
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import checkFasta
import readFasta
import saveCode

USAGE = """
USAGE:
	python PAAC.py input.fasta <lambda> <output>

	input.fasta:      the input protein sequence file in fasta format.
	lambda:           the lambda value, integer, defaule: 30
	output:           the encoding file, default: 'encodings.tsv'
"""


def Rvalue(
    aa1: str, aa2: str, AADict: Mapping[str, int], Matrix: Sequence[Sequence[float]]
) -> float:
    """Rvalue.

    :param aa1: Aa1.
    :type aa1: str
    :param aa2: Aa2.
    :type aa2: str
    :param AADict: Aadict.
    :type AADict: Mapping[str, int]
    :param Matrix: Matrix.
    :type Matrix: Sequence[Sequence[float]]
    :return: Result of this operation.
    :rtype: float
    """
    return sum(
        [
            (Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2
            for i in range(len(Matrix))
        ]
    ) / len(Matrix)


def PAAC(
    fastas: Sequence[Sequence[str]],
    lambdaValue: int = 30,
    w: float = 0.05,
    **kw: object,
) -> int | list[object]:
    """Paac.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param lambdaValue: Lambdavalue.
    :type lambdaValue: int
    :param w: W.
    :type w: float
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: int | list[object]
    """
    if checkFasta.minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
        print(
            "Error: all the sequence length should be larger than the lambdaValue+1: "
            + str(lambdaValue + 1)
            + "\n\n"
        )
        return 0

    dataFile = (
        re.sub("codes$", "", os.path.split(os.path.realpath(__file__))[0])
        + r"\data\PAAC.txt"
        if platform.system() == "Windows"
        else re.sub("codes$", "", os.path.split(os.path.realpath(__file__))[0])
        + "/data/PAAC.txt"
    )
    with open(dataFile) as f:
        records = f.readlines()
    AA = "".join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != "" else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = ["#"]
    for aa in AA:
        header.append("Xc1." + aa)
    for n in range(1, lambdaValue + 1):
        header.append("Xc2.lambda" + str(n))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum(
                    [
                        Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1)
                        for j in range(len(sequence) - n)
                    ]
                )
                / (len(sequence) - n)
            )
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return encodings


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(USAGE)
        sys.exit(1)
    fastas = readFasta.readFasta(sys.argv[1])
    lambdaValue = int(sys.argv[2]) if len(sys.argv) >= 3 else 30
    output = sys.argv[3] if len(sys.argv) >= 4 else "encoding.tsv"
    encodings = PAAC(fastas, lambdaValue)
    saveCode.savetsv(encodings, output)
