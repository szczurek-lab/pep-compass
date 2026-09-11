"""Kernel implementations used by optimization policies."""

import torch
import gpytorch


class TanimotoSimilarityKernel(gpytorch.kernels.Kernel):
    """
    A class computing the Tanimoto similarity coefficient between input data points.
    """

    is_stationary = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, eps=1e-6, **params):
        if last_dim_is_batch:
            raise NotImplementedError("last_dim_is_batch=True is not supported.")

        # Ensure inputs are at least 3D: [batch_shape x n x d]
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)

        x1_eq_x2 = torch.equal(x1, x2)

        x1s = torch.sum(x1 ** 2, dim=-1)  # shape: [batch, n]
        x2s = torch.sum(x2 ** 2, dim=-1)  # shape: [batch, m]

        if diag:
            if x1_eq_x2:
                res = torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                return res
            else:
                product = torch.mul(x1, x2).sum(dim=1)
                denominator = torch.add(x2s, x1s) - product
        else:
            # Inner product between all pairs: [batch, n, d] @ [batch, d, m] -> [batch, n, m]
            product = torch.matmul(x1, x2.transpose(-2, -1))
            # Broadcasted denominator: [batch, n, m]
            denominator = x1s.unsqueeze(-1) + x2s.unsqueeze(-2) - product
            res = (product + eps) / (denominator + eps)

        return res.squeeze(0) if res.shape[0] == 1 else res
