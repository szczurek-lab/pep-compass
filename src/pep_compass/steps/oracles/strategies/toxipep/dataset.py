from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from atom_feature import convert_to_graph_channel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


# 残基到索引的映射
Pep_residue2idx = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "A": 3,  # Alanine
    "C": 4,  # Cysteine
    "D": 5,  # Aspartic acid
    "E": 6,  # Glutamic acid
    "F": 7,  # Phenylalanine
    "G": 8,  # Glycine
    "H": 9,  # Histidine
    "I": 10,  # Isoleucine
    "K": 11,  # Lysine
    "L": 12,  # Leucine
    "M": 13,  # Methionine
    "N": 14,  # Asparagine
    "P": 15,  # Proline
    "Q": 16,  # Glutamine
    "R": 17,  # Arginine
    "S": 18,  # Serine
    "T": 19,  # Threonine
    "V": 20,  # Valine
    "W": 21,  # Tryptophan
    "Y": 22,  # Tyrosine
}


def transform_Pep_to_index(
    sequences: Sequence[str], residue2idx: Mapping[str, int]
) -> list[object]:
    """Transform pep to index.

    :param sequences: Peptide sequences in input order.
    :type sequences: Sequence[str]
    :param residue2idx: Residue2idx.
    :type residue2idx: Mapping[str, int]
    :return: Result of this operation.
    :rtype: list[object]
    """
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq]
        token_index.append(seq_id)
    return token_index


def pad_sequence(
    token_list: Sequence[Sequence[int]], max_len: int = 51
) -> list[list[int]]:
    """Pad sequence.

    :param token_list: Token list.
    :type token_list: Sequence[Sequence[int]]
    :param max_len: Max len.
    :type max_len: int
    :return: Result of this operation.
    :rtype: list[list[int]]
    """
    data = []
    for tokens in token_list:
        seq = [Pep_residue2idx["[CLS]"]] + tokens
        n_pad = max_len - len(seq)
        seq.extend([Pep_residue2idx["[PAD]"]] * max(n_pad, 0))
        data.append(seq[:max_len])
    return data


def load_data_from_fasta(
    file_path: str | Path, max_len: int = 51
) -> tuple[object, list[object], list[object]]:
    """Load data from fasta.

    :param file_path: File path.
    :type file_path: str | Path
    :param max_len: Max len.
    :type max_len: int
    :return: Result of this operation.
    :rtype: tuple[object, list[object], list[object]]
    """
    sequences = []
    labels = []
    with open(file_path, "r") as f:
        current_seq = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]

                label = 1 if header.lower().startswith("pos") else 0
                labels.append(label)
                current_seq = ""
                sequences.append(current_seq)
            else:
                sequences[-1] += line

    indexed_sequences = transform_Pep_to_index(sequences, Pep_residue2idx)
    padded_sequences = pad_sequence(indexed_sequences, max_len=max_len)

    graph_features = [convert_to_graph_channel(seq) for seq in sequences]

    return padded_sequences, graph_features, labels


class MyDataSet(Data.Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        graph_features: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param input_ids: Input ids.
        :type input_ids: torch.Tensor
        :param graph_features: Graph features.
        :type graph_features: torch.Tensor | np.ndarray
        :param labels: Labels.
        :type labels: torch.Tensor | np.ndarray
        :rtype: None
        """
        self.input_ids = input_ids
        self.graph_features = graph_features
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of contained elements.

        :return: Result of this operation.
        :rtype: int
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the element selected by the requested index.

        :param idx: Idx.
        :type idx: int
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.graph_features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
