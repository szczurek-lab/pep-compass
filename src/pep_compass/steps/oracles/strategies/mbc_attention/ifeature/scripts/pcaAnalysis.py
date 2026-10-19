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


pPath = re.sub(r"scripts$", "", os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
import argparse

import matplotlib as mpl
import numpy as np
from clusters import readCode
from sklearn.decomposition import PCA

mpl.use("Agg")
import matplotlib.pyplot as plt


def pacAnalysis(
    encodings: Sequence[Sequence[str | float | int]], n_components: int = 2
) -> list[list[str | float]]:
    """Pac analysis.

    :param encodings: Feature table including the identifiers and numeric columns expected by this operation.
    :type encodings: Sequence[Sequence[str | float | int]]
    :param n_components: N components.
    :type n_components: int
    :return: Result of this operation.
    :rtype: list[list[str | float]]
    """
    encodings = np.array(encodings)[1:]
    data = encodings[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)
    newData = PCA(n_components=n_components).fit_transform(data)
    pca = []
    for i in range(len(data)):
        pca.append([encodings[i][0]] + list(newData[i]))
    return pca


def savePCA(pca: Sequence[Sequence[str | float | int]], file: str = "PCA.txt") -> None:
    """Save pca.

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


def pcaPlot(
    pca: Sequence[Sequence[str | float | int]],
    label: Sequence[str | int],
    file: str = "pca.png",
) -> None:
    """Pca plot.

    :param pca: Pca.
    :type pca: Sequence[Sequence[str | float | int]]
    :param label: Label.
    :type label: Sequence[str | int]
    :param file: File.
    :type file: str
    :rtype: None
    """
    pca = np.array(pca)[:]
    data = pca[:, 1:]
    shape = data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)

    if len(label) == 0:
        plt.scatter(data[:, 0], data[:, 1], 20, c="r")
    else:
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
    parser.add_argument("--file", required=True, help="data file for PCA")
    parser.add_argument("--out", help="output file of PCA analysis.")
    parser.add_argument("--ncomponents", help="number of n components, default 2")
    parser.add_argument("--label", help="optimal, the label file for the input file")
    args = parser.parse_args()
    output = args.out if args.out != None else "PCA.txt"
    n_components = int(args.ncomponents) if args.ncomponents != None else 2
    data = readCode.readCode(args.file)
    label = []
    if args.label != None:
        with open(args.label) as f:
            records = f.readlines()
        for line in records:
            array = line.strip().split() if line.strip() != "" else None
            label.append(array[1])

    pca = pacAnalysis(data, n_components)
    savePCA(pca, output)
    pcaPlot(pca, label, "%s.png" % output)
