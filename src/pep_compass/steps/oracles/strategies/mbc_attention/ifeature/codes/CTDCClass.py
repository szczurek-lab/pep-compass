from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import os
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import readFasta
import saveCode

USAGE = """
USAGE:
	python CTDCClass.py input.fasta output amino_acids_group_1 amino_acids_group_2 ... amino_acids_group_N

	input.fasta:                 the input protein sequence file in fasta format.
	output:                      the encoding file.
	amino_acids_group_x          the amino acids groups.

EXAMPLE:
	python CTDCClass.py example/test-protein.txt CTDCClass.tsv RKEDQN GASTPHY CLVIMFW
"""


def Count(seq1: str, seq2: str) -> int:
    """Count.

    :param seq1: Seq1.
    :type seq1: str
    :param seq2: Seq2.
    :type seq2: str
    :return: Result of this operation.
    :rtype: int
    """
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum


def CTDCClass(fastas: Sequence[Sequence[str]], groups: Sequence[str]) -> list[object]:
    """Ctdcclass.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param groups: Groups.
    :type groups: Sequence[str]
    :return: Result of this operation.
    :rtype: list[object]
    """
    encodings = []
    header = ["#"]
    for g in range(len(groups)):
        header.append("g." + str(g + 1))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        for group in groups:
            code.append(Count(group, sequence) / len(sequence))
        encodings.append(code)
    return encodings


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(USAGE)
        sys.exit(1)

    groups = sys.argv[3:]
    myStr = "".join(groups)
    myStr = re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "", myStr)
    if len(myStr) != 20 or len(set(myStr)) != 20:
        print(
            "\nERROR: The amino acid must be no-repeat in each groups and the sum is 20!\n"
        )
    fastas = readFasta.readFasta(sys.argv[1])
    encodings = CTDCClass(fastas, groups)
    saveCode.savetsv(encodings, sys.argv[2])
