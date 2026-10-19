from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def GAAC(fastas: Sequence[Sequence[str]], **kw: object) -> list[object]:
    """Gaac.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: list[object]
    """
    group = {
        "alphatic": "GAVLMI",
        "aromatic": "FYW",
        "postivecharge": "KRH",
        "negativecharge": "DE",
        "uncharge": "STCPNQ",
    }

    groupKey = group.keys()

    encodings = []
    header = ["#"]
    for key in groupKey:
        header.append(key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        code = [name]
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        for key in groupKey:
            code.append(myDict[key] / len(sequence))
        encodings.append(code)

    return encodings
