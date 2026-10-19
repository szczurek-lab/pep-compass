from __future__ import annotations

from typing import TYPE_CHECKING

#!/usr/bin/env python
# _*_coding:utf-8_*_
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Sequence


def dbscan(
    encodings: Sequence[Sequence[str | float | int]], **kw: object
) -> tuple[int, str] | tuple[list[object], str]:
    """Dbscan.

    :param encodings: Feature table including the identifiers and numeric columns expected by this operation.
    :type encodings: Sequence[Sequence[str | float | int]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: tuple[int, str] | tuple[list[object], str]
    """
    if kw["sof"] == "sample":
        encodings = np.array(encodings)[1:]
    else:
        encodings = np.array(encodings).T[1:]

    if len(encodings) < 5:
        return 0, "sample number should be greater than 5."

    data = encodings[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)

    e = ""
    dataNew = StandardScaler().fit_transform(data)
    db = DBSCAN().fit(dataNew)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    res = []
    for i in range(len(data)):
        res.append([encodings[i][0], labels[i]])
    return res, e
