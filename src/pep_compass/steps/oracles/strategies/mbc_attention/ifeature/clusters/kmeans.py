from __future__ import annotations

from typing import TYPE_CHECKING

#!/usr/bin/env python
# _*_coding:utf-8_*_
import numpy as np
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from collections.abc import Sequence


def kmeans(
    encodings: Sequence[Sequence[str | float | int]], **kw: object
) -> tuple[int, str] | tuple[list[object], str]:
    """Kmeans.

    :param encodings: Feature table including the identifiers and numeric columns expected by this operation.
    :type encodings: Sequence[Sequence[str | float | int]]
    :param kw: Additional feature-encoder options.
    :type kw: object
    :return: Result of this operation.
    :rtype: tuple[int, str] | tuple[list[object], str]
    """
    nclusters = int(kw["nclusters"]) if kw["nclusters"] != None else 3
    if kw["sof"] == "sample":
        encodings = np.array(encodings)[1:]
    else:
        encodings = np.array(encodings).T[1:]

    if len(encodings) < nclusters:
        return 0, "sample number should be greater than n_clusters."

    data = encodings[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)
    e = ""
    try:
        cluster_pred = KMeans(n_clusters=nclusters).fit_predict(data)
    except ValueError as e:
        print(e)
        return 0, e

    res = []
    for i in range(len(data)):
        res.append([encodings[i][0], cluster_pred[i]])
    return res, e
