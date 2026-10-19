"""HydrAMP encoder and decoder network architecture."""

from __future__ import annotations

import torch
from torch import nn


class HydrAMPGRU(nn.Module):
    """Reproduce the recurrent HydrAMP GRU cell and sequence unrolling."""

    def __init__(
        self,
        units: int = 66,
        input_units: int = 66,
        output_len: int = 25,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize this object with the declared configuration.

        :param units: Units.
        :type units: int
        :param input_units: Input units.
        :type input_units: int
        :param output_len: Output len.
        :type output_len: int
        :param device: Computation device accepted by the numerical backend.
        :type device: str | torch.device
        :rtype: None
        """
        super().__init__()
        self.output_len = output_len
        self.units = units
        self.input_units = input_units
        self.device = device
        self.kernel = torch.nn.Parameter(
            torch.zeros(size=(input_units, units * 3), device=device)
        )
        self.recurrent_kernel = torch.nn.Parameter(
            torch.zeros(size=(units, units * 3), device=device)
        )
        self.bias = torch.nn.Parameter(torch.zeros(size=(units * 3,), device=device))

    def cell_forward(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one recurrent cell state.

        :param inputs: Inputs.
        :type inputs: torch.Tensor
        :param state: State.
        :type state: torch.Tensor
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        h_tm1 = state

        input_bias = self.bias

        matrix_x = torch.matmul(inputs, self.kernel)
        matrix_x = matrix_x + input_bias

        x_z, x_r, x_h = torch.split(matrix_x, self.units, dim=-1)

        matrix_inner = torch.matmul(h_tm1, self.recurrent_kernel[: self.units * 2])

        recurrent_z, recurrent_r, recurrent_h = torch.split(
            matrix_inner, self.units, dim=-1
        )

        z = torch.sigmoid(x_z + recurrent_z)
        r = torch.sigmoid(x_r + recurrent_r)

        recurrent_h = torch.matmul(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units :]
        )

        hh = torch.tanh(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        new_state = h
        return h, new_state

    def forward(
        self, input_: torch.Tensor | None, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Generate a fixed-length recurrent output from an initial state.

        :param input_: Input .
        :type input_: torch.Tensor | None
        :param state: State.
        :type state: torch.Tensor | None
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        if input_ is None:
            input_ = torch.zeros((state.shape[0], self.input_units), device=self.device)
        if state is None:
            state = torch.zeros((input_.shape[0], self.units), device=self.device)
        current_output = input_
        current_state = state
        outputs = []
        for _ in range(self.output_len):
            current_output, current_state = self.cell_forward(
                current_output,
                current_state,
            )
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)

    def forward_on_sequence(
        self, input_: torch.Tensor | None, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the recurrent cell to every input sequence position.

        :param input_: Input .
        :type input_: torch.Tensor | None
        :param state: State.
        :type state: torch.Tensor | None
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        if input_ is None:
            input_ = torch.zeros((state.shape[0], self.input_units), device=self.device)
        if state is None:
            state = torch.zeros((input_.shape[0], self.units), device=self.device)
        current_state = state
        outputs = []
        for i in range(self.output_len):
            current_output, current_state = self.cell_forward(
                input_[:, i],
                current_state,
            )
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)


class HydrAMPDecoder(nn.Module):
    """Decode conditioned latent positions into peptide-token logits."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        """Initialize this object with the declared configuration.

        :param device: Computation device accepted by the numerical backend.
        :type device: str | torch.device
        :rtype: None
        """
        super().__init__()
        self.device = device
        self.gru = HydrAMPGRU(device=self.device)
        self.lstm = torch.nn.LSTM(68, 100, batch_first=True).to(device)
        self.dense = torch.nn.Linear(102, 21).to(device)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = True,
        gumbel_temperature: float = 0.001,
    ) -> torch.Tensor:
        """Return decoder logits or a Gumbel-softmax token sample.

        :param x: X.
        :type x: torch.Tensor
        :param return_logits: Return logits.
        :type return_logits: bool
        :param gumbel_temperature: Gumbel temperature.
        :type gumbel_temperature: float
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        x = x.to(self.device)
        gru_output = self.gru(None, x)
        condition = x[:, -2:]
        condition_repeat = condition.unsqueeze(1).repeat(1, 25, 1)
        gru_output_with_condition = torch.concat([gru_output, condition_repeat], dim=-1)
        lstm_output = self.lstm(gru_output_with_condition)[0]
        lstm_output_with_condition = torch.concat(
            [lstm_output, condition_repeat], dim=-1
        )
        dense_output = self.dense(lstm_output_with_condition)
        if return_logits:
            return dense_output
        return torch.nn.functional.gumbel_softmax(dense_output, tau=gumbel_temperature)


class HydrAMPEncoder(nn.Module):
    """Encode peptide tokens into HydrAMP latent mean and scale vectors."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        """Initialize this object with the declared configuration.

        :param device: Computation device accepted by the numerical backend.
        :type device: str | torch.device
        :rtype: None
        """
        super().__init__()
        self.device = device
        self.embedding = torch.nn.Embedding(
            num_embeddings=21, embedding_dim=100, device=self.device
        )
        self.gru1_f = HydrAMPGRU(input_units=100, units=128, device=self.device)
        self.gru1_r = HydrAMPGRU(input_units=100, units=128, device=self.device)
        self.gru2_f = HydrAMPGRU(input_units=256, units=128, device=self.device)
        self.gru2_r = HydrAMPGRU(input_units=256, units=128, device=self.device)
        self.mean_linear = torch.nn.Linear(256, 64, device=self.device)
        self.std_linear = torch.nn.Linear(256, 64, device=self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and scale tensors for a token batch.

        :param x: X.
        :type x: torch.Tensor
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        x = x.to(self.device)
        embeddings = self.embedding(x)
        gru1_f_output = self.gru1_f.forward_on_sequence(embeddings)
        gru1_r_output = self.gru1_r.forward_on_sequence(torch.flip(embeddings, (1,)))
        gru_1_output = torch.concat(
            [gru1_f_output, torch.flip(gru1_r_output, (1,))], dim=-1
        )
        gru2_f_output = self.gru2_f.forward_on_sequence(gru_1_output)
        gru2_r_output = self.gru2_r.forward_on_sequence(torch.flip(gru_1_output, (1,)))
        gru_2_output = torch.concat(
            [gru2_f_output[:, -1], gru2_r_output[:, -1]], dim=-1
        )
        mean = self.mean_linear(gru_2_output)
        std = self.std_linear(gru_2_output)
        return mean, std
