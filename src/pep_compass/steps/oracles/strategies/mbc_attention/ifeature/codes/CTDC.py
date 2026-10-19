from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def CTDC(fastas: Sequence[Sequence[str]], **kw: object) -> list[object]:
    """Ctdc.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: list[object]
    """
    group1 = {
        "hydrophobicity_PRAM900101": "RKEDQN",
        "hydrophobicity_ARGP820101": "QSTNGDE",
        "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
        "hydrophobicity_PONP930101": "KPDESNQT",
        "hydrophobicity_CASG920101": "KDEQPSRNTG",
        "hydrophobicity_ENGD860101": "RDKENQHYP",
        "hydrophobicity_FASG890101": "KERSQD",
        "normwaalsvolume": "GASTPDC",
        "polarity": "LIFWCMVY",
        "polarizability": "GASDT",
        "charge": "KR",
        "secondarystruct": "EALMQKRH",
        "solventaccess": "ALFCGIVW",
    }
    group2 = {
        "hydrophobicity_PRAM900101": "GASTPHY",
        "hydrophobicity_ARGP820101": "RAHCKMV",
        "hydrophobicity_ZIMJ680101": "HMCKV",
        "hydrophobicity_PONP930101": "GRHA",
        "hydrophobicity_CASG920101": "AHYMLV",
        "hydrophobicity_ENGD860101": "SGTAW",
        "hydrophobicity_FASG890101": "NTPG",
        "normwaalsvolume": "NVEQIL",
        "polarity": "PATGS",
        "polarizability": "CPNVEQIL",
        "charge": "ANCQGHILMFPSTWYV",
        "secondarystruct": "VIYCWFT",
        "solventaccess": "RKQEND",
    }
    group3 = {
        "hydrophobicity_PRAM900101": "CLVIMFW",
        "hydrophobicity_ARGP820101": "LYPFIW",
        "hydrophobicity_ZIMJ680101": "LPFYI",
        "hydrophobicity_PONP930101": "YMFWLCVI",
        "hydrophobicity_CASG920101": "FIWC",
        "hydrophobicity_ENGD860101": "CVLIMF",
        "hydrophobicity_FASG890101": "AYHWVMFLIC",
        "normwaalsvolume": "MHKFRYW",
        "polarity": "HQRKNED",
        "polarizability": "KMHFRYW",
        "charge": "DE",
        "secondarystruct": "GNPSD",
        "solventaccess": "MSPTHY",
    }

    groups = [group1, group2, group3]
    property = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )

    encodings = []
    header = ["#"]
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + ".G" + str(g))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        for p in property:
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)
    return encodings
