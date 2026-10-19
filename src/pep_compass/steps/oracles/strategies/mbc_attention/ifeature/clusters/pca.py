from __future__ import annotations

from typing import TYPE_CHECKING

#!/usr/bin/env python
# _*_coding:utf-8_*_
import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from collections.abc import Sequence


def pca(
    encodings: Sequence[Sequence[str | float | int]], n_components: int = 2
) -> np.ndarray:
    """Pca.

    :param encodings: Feature table including the identifiers and numeric columns expected by this operation.
    :type encodings: Sequence[Sequence[str | float | int]]
    :param n_components: N components.
    :type n_components: int
    :return: Result of this operation.
    :rtype: np.ndarray
    """
    encodings = np.array(encodings)
    data = encodings[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)
    newData = PCA(n_components=n_components).fit_transform(data)
    pca = []
    for i in range(len(data)):
        pca.append([encodings[i][0]] + list(newData[i]))
    return np.array(pca)
