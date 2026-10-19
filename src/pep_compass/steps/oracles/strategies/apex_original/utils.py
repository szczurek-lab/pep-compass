from __future__ import annotations

import copy
import csv
import json
import math
import os
import os.path
import pickle
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from torch.autograd import Variable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def make_vocab() -> tuple[dict[object, object], dict[object, object]]:
    # 0: pad
    # 1: start
    # 2: end

    """Make vocab.

    :return: Result of this operation.
    :rtype: tuple[dict[object, object], dict[object, object]]
    """
    word2idx = {}
    idx2word = {}

    word2idx["0"] = 0
    word2idx["1"] = 1
    word2idx["2"] = 2

    word2idx["A"] = 3
    word2idx["C"] = 4
    word2idx["D"] = 5
    word2idx["E"] = 6
    word2idx["F"] = 7
    word2idx["G"] = 8
    word2idx["H"] = 9
    word2idx["I"] = 10
    word2idx["K"] = 11
    word2idx["L"] = 12
    word2idx["M"] = 13
    word2idx["N"] = 14
    word2idx["P"] = 15
    word2idx["Q"] = 16
    word2idx["R"] = 17
    word2idx["S"] = 18
    word2idx["T"] = 19
    word2idx["V"] = 20
    word2idx["W"] = 21
    word2idx["Y"] = 22

    for key, value in word2idx.items():
        idx2word[value] = key

    return word2idx, idx2word


def AAindex(
    path: str | Path, word2idx: Mapping[str, int]
) -> tuple[np.ndarray, dict[object, object]]:
    """Aaindex.

    :param path: Input or output path used by this operation.
    :type path: str | Path
    :param word2idx: Word2idx.
    :type word2idx: Mapping[str, int]
    :return: Result of this operation.
    :rtype: tuple[np.ndarray, dict[object, object]]
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        AAindex_dict = {}
        AAindex_matrix = []
        skip = 1
        for row in reader:
            if skip == 1:
                skip = 0
                header = np.array(row)[1:].tolist()
                continue
            tmp = []
            for j in np.array(row)[1:]:
                try:
                    tmp.append(float(j))
                except:
                    tmp.append(0)
            AAindex_matrix.append(np.array(tmp))

        dim = np.shape(AAindex_matrix)[0]
        AAindex_matrix = np.array(AAindex_matrix)
        for i in range(len(header)):
            AAindex_dict[header[i]] = AAindex_matrix[:, i]

    # print (AAindex_matrix)
    emb = np.zeros((len(word2idx), dim))
    for key, value in word2idx.items():
        if key in AAindex_dict:
            emb[value] = AAindex_dict[key]
        else:
            pass
    return emb, AAindex_dict


def onehot_encoding(
    seq_list_: Sequence[str], max_len: int, word2idx: Mapping[str, int]
) -> np.ndarray:
    # 0: pad
    # 1: start
    # 2: end
    """Onehot encoding.

    :param seq_list_: Seq list .
    :type seq_list_: Sequence[str]
    :param max_len: Max len.
    :type max_len: int
    :param word2idx: Word2idx.
    :type word2idx: Mapping[str, int]
    :return: Result of this operation.
    :rtype: np.ndarray
    """
    seq_list = [i for i in seq_list_]
    X = np.zeros((len(seq_list), max_len)).astype(int)

    AA_mask = []
    nonAA_mask = []

    for i in range(len(seq_list)):
        if len(seq_list[i]) >= max_len - 2:
            a_seq = "1" + seq_list[i][: max_len - 2].upper() + "2"
        else:
            a_seq = "1" + seq_list[i].upper() + "2"

        if len(a_seq) > max_len:
            iter_num = max_len
        else:
            iter_num = len(a_seq)

        for j in range(iter_num):
            if a_seq[j] not in word2idx:
                continue
            else:
                X[i, j] = word2idx[a_seq[j]]

        tmp = np.zeros(max_len)
        tmp[1 : iter_num + 1] = 1
        AA_mask.append(tmp.astype(int))
        nonAA_mask.append((1 - tmp).astype(int))

    return np.array(X)  # , np.array(AA_mask), np.array(nonAA_mask)
