from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def AAC(fastas: Sequence[Sequence[str]], **kw: object) -> list[object]:
    """Aac.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: list[object]
    """
    AA = kw["order"] if kw["order"] != None else "ACDEFGHIKLMNPQRSTVWY"
    # AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = ["#"]
    for i in AA:
        header.append(i)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub("-", "", i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = [name]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings


if __name__ == "__main__":
    from readFasta import readFasta

    path = "/home/mb95537/acp-design/ion_channels/uniprot-preprocessed-dataset/calcium.fasta"
    fastas = readFasta(path)
    encdn = AAC(fastas, order=None)
    import pandas as pd

    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    print("%s's feature number: %d" % (AAC.__name__, len(df.columns)))
    # add class
    df["default"] = [1] * len(fastas)
