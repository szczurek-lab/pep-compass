from __future__ import annotations

#!/usr/bin/env python
# _*_coding:utf-8_*_
import os
import re
import sys
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


pPath = re.sub(r"scripts$", "", os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
import argparse

import matplotlib as mpl
import numpy as np
from clusters import readCode
from sklearn.decomposition import LatentDirichletAllocation

mpl.use("Agg")
import matplotlib.pyplot as plt


def ldaAnalysis(
    encodings: Sequence[Sequence[str | float | int]],
    labelFile: str | Path,
    n_components: int = 2,
) -> list[list[str | float]] | tuple[int, IndexError]:
    """Lda analysis.

    :param encodings: Feature table including the identifiers and numeric columns expected by this operation.
    :type encodings: Sequence[Sequence[str | float | int]]
    :param labelFile: Labelfile.
    :type labelFile: str | Path
    :param n_components: N components.
    :type n_components: int
    :return: Result of this operation.
    :rtype: list[list[str | float]] | tuple[int, IndexError]
    """
    with open(labelFile) as f:
        records = f.readlines()[1:]
    myDict = {}
    try:
        for i in records:
            array = i.rstrip().split() if i.strip() != "" else None
            myDict[array[0]] = int(array[1])
    except IndexError as e:
        return 0, e

    labels = []
    for i in encodings:
        labels.append(myDict.get(i[0], 0))

    encodings = np.array(encodings)[1:]
    data = encodings[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)
    newData = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="batch",
        learning_offset=50.0,
        random_state=0,
    ).fit_transform(data, labels)
    lda = []
    for i in range(len(data)):
        lda.append([encodings[i][0]] + list(newData[i]))
    return lda


def saveLDA(pca: Sequence[Sequence[str | float | int]], file: str = "LDA.txt") -> None:
    """Save lda.

    :param pca: Pca.
    :type pca: Sequence[Sequence[str | float | int]]
    :param file: File.
    :type file: str
    :rtype: None
    """
    with open(file, "w") as f:
        f.write("protein\t")
        for i in range(1, len(pca[0])):
            f.write("pc." + str(i) + "\t")
        f.write("\n")
        for i in pca:
            for j in i:
                f.write(str(j) + "\t")
            f.write("\n")
    return None


def ldaPlot(
    lda: Sequence[Sequence[str | float | int]],
    label: Sequence[str | int],
    file: str = "lda.png",
) -> None:
    """Lda plot.

    :param lda: Lda.
    :type lda: Sequence[Sequence[str | float | int]]
    :param label: Label.
    :type label: Sequence[str | int]
    :param file: File.
    :type file: str
    :rtype: None
    """
    lda = np.array(lda)[:]
    data = lda[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)

    df = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1], "L": label})
    mySet = set(label)
    for l in mySet:
        newData = df.loc[df.loc[:, "L"] == l, :]
        plt.scatter(np.array(newData.X), np.array(newData.Y), 20, label="%s" % l)
        plt.legend(loc="best")
    plt.savefig(file)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="it's usage tip.", description="PCA and plot"
    )
    parser.add_argument("--file", required=True, help="data file for LDA")
    parser.add_argument("--label", required=True, help="sample label file")
    parser.add_argument("--out", help="output file of LDA analysis.")
    parser.add_argument("--ncomponents", help="number of n components, default 2")
    args = parser.parse_args()
    n_components = int(args.ncomponents) if args.ncomponents != None else 2
    ldaFile = args.out if args.out != None else "lda.txt"

    data = readCode.readCode(args.file)
    label = []
    if args.label != None:
        with open(args.label) as f:
            records = f.readlines()
        for line in records:
            array = line.strip().split() if line.strip() != "" else None
            label.append(array[1])

    lda = ldaAnalysis(data, args.label, n_components)
    saveLDA(lda, ldaFile)
    ldaPlot(lda, label, "%s.png" % ldaFile)
