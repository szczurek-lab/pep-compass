from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def checkFasta(fastas: Sequence[Sequence[str]]) -> bool:
    """Check fasta.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :return: Result of this operation.
    :rtype: bool
    """
    status = True
    lenList = set()
    for i in fastas:
        lenList.add(len(i[1]))
    if len(lenList) == 1:
        return True
    else:
        return False


def minSequenceLength(fastas: Sequence[Sequence[str]]) -> int:
    """Min sequence length.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :return: Result of this operation.
    :rtype: int
    """
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen


def minSequenceLengthWithNormalAA(fastas: Sequence[Sequence[str]]) -> int:
    """Min sequence length with normal aa.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :return: Result of this operation.
    :rtype: int
    """
    minLen = 10000
    for i in fastas:
        if minLen > len(re.sub("-", "", i[1])):
            minLen = len(re.sub("-", "", i[1]))
    return minLen
