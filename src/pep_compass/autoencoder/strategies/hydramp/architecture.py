"""HydrAMP encoder and decoder network architecture."""

import torch
from torch import nn


class HydrAMPGRU(nn.Module):
    """Reproduce the recurrent HydrAMP GRU cell and sequence unrolling."""

    def __init__(
        self,
        units=66,
        input_units=66,
        output_len=25,
        device='cpu'
    ):
        super().__init__()
        self.output_len = output_len
        self.units = units
        self.input_units = input_units
        self.device = device
        self.kernel = torch.nn.Parameter(torch.zeros(size=(input_units, units * 3), device=device))
        self.recurrent_kernel = torch.nn.Parameter(torch.zeros(size=(units, units * 3), device=device))
        self.bias = torch.nn.Parameter(torch.zeros(size=(units * 3,), device=device))

    def cell_forward(self, inputs, state):
        """Advance one recurrent cell state."""
        h_tm1 = state

        input_bias = self.bias

        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

        matrix_x = torch.matmul(inputs, self.kernel)
        matrix_x = matrix_x + input_bias

        x_z, x_r, x_h = torch.split(matrix_x, self.units, dim=-1)

        matrix_inner = torch.matmul(h_tm1, self.recurrent_kernel[:self.units*2])

        recurrent_z, recurrent_r, recurrent_h = torch.split(
            matrix_inner, self.units, dim=-1
        )

        z = torch.sigmoid(x_z + recurrent_z)
        r = torch.sigmoid(x_r + recurrent_r)

        recurrent_h = torch.matmul(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units:])

        hh = torch.tanh(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        new_state = h
        return h, new_state

    def forward(self, input_, state=None):
        """Generate a fixed-length recurrent output from an initial state."""
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

    def forward_on_sequence(self, input_, state=None):
        """Apply the recurrent cell to every input sequence position."""
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
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.gru = HydrAMPGRU(device=self.device)
        self.lstm = torch.nn.LSTM(68, 100, batch_first=True).to(device)
        self.dense = torch.nn.Linear(102, 21).to(device)
        
    def forward(self, x, return_logits=True, gumbel_temperature=0.001):
        """Return decoder logits or a Gumbel-softmax token sample."""
        x = x.to(self.device)
        gru_output = self.gru(None, x)
        condition = x[:, -2:]
        condition_repeat = condition.unsqueeze(1).repeat(1, 25, 1)
        gru_output_with_condition = torch.concat([gru_output, condition_repeat], dim=-1)
        lstm_output = self.lstm(gru_output_with_condition)[0]
        lstm_output_with_condition = torch.concat([lstm_output, condition_repeat], dim=-1)
        dense_output = self.dense(lstm_output_with_condition)
        if return_logits:
            return dense_output
        return torch.nn.functional.gumbel_softmax(dense_output, tau=gumbel_temperature)
    
    
class HydrAMPEncoder(nn.Module):
    """Encode peptide tokens into HydrAMP latent mean and scale vectors."""
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = torch.nn.Embedding(
            num_embeddings=21,
            embedding_dim=100,
            device=self.device
        )
        self.gru1_f = HydrAMPGRU(input_units=100, units=128, device=self.device)
        self.gru1_r = HydrAMPGRU(input_units=100, units=128, device=self.device)
        self.gru2_f = HydrAMPGRU(input_units=256, units=128, device=self.device)
        self.gru2_r = HydrAMPGRU(input_units=256, units=128, device=self.device)
        self.mean_linear = torch.nn.Linear(256, 64, device= self.device)
        self.std_linear = torch.nn.Linear(256, 64, device= self.device)
        
    def forward(self, x):
        """Return latent mean and scale tensors for a token batch."""
        x = x.to(self.device)
        embeddings = self.embedding(x)
        gru1_f_output = self.gru1_f.forward_on_sequence(embeddings)
        gru1_r_output = self.gru1_r.forward_on_sequence(torch.flip(embeddings, (1,)))
        gru_1_output = torch.concat([gru1_f_output, torch.flip(gru1_r_output, (1,))], dim=-1)
        gru2_f_output = self.gru2_f.forward_on_sequence(gru_1_output)
        gru2_r_output = self.gru2_r.forward_on_sequence(torch.flip(gru_1_output, (1,)))
        gru_2_output = torch.concat([gru2_f_output[:, -1], gru2_r_output[:, -1]], dim=-1)
        mean = self.mean_linear(gru_2_output)
        std = self.std_linear(gru_2_output)
        return mean, std
