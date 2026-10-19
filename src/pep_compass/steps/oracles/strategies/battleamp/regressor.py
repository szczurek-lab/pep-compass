from __future__ import annotations

# the MIC predictor nn
import numpy as np
import tensorflow as tf
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    ZeroPadding1D,
)
from tensorflow import keras


def cnn_regressor(max_sequence_length: int, num_tokens: int) -> keras.Model:
    """CNN neural network for MIC prediction from peptide sequence

    Returns:
        Keras sequential model: CNN regressor

    :param max_sequence_length: Explicit input sequence length.
    :type max_sequence_length: int
    :param num_tokens: Explicit token feature width.
    :type num_tokens: int
    :return: Result of this operation.
    :rtype: keras.Model
    """
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(5, input_shape=(max_sequence_length, num_tokens)))
    model.add(Conv1D(64, kernel_size=5, strides=1, activation="relu"))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def rnn_regressor(max_sequence_length: int, num_tokens: int) -> keras.Model:
    """RNN neural network for MIC prediction from peptide sequence

    Returns:
        Keras sequential model: RNN regressor

    :param max_sequence_length: Explicit input sequence length.
    :type max_sequence_length: int
    :param num_tokens: Explicit token feature width.
    :type num_tokens: int
    :return: Result of this operation.
    :rtype: keras.Model
    """
    model = keras.models.Sequential()
    model.add(LSTM(500, input_shape=(max_sequence_length, num_tokens)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model
