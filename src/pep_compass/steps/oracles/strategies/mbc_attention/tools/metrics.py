from __future__ import annotations

from typing import TYPE_CHECKING

from tools import *
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    L1Loss,
    Linear,
    MaxPool2d,
    Module,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
)
from torch.optim import SGD, Adam

if TYPE_CHECKING:
    import pandas as pd
    import torch


def r2_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R2 loss.

    :param output: Output.
    :type output: torch.Tensor
    :param target: Target.
    :type target: torch.Tensor
    :return: Result of this operation.
    :rtype: torch.Tensor
    """
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def PCC(
    output: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pcc.

    :param output: Output.
    :type output: torch.Tensor
    :param target: Target.
    :type target: torch.Tensor
    :return: Result of this operation.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    eps = 1e-6
    x_ave, y_ave = torch.mean(target), torch.mean(output)
    vx, vy = target - x_ave, output - y_ave
    x_std, y_std = torch.std(target), torch.std(output)
    x_var, y_var = torch.var(target), torch.var(output)
    pcc_value = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2) + eps) * torch.sqrt(torch.sum(vy**2) + eps)
    )
    ccc_value = (2 * pcc_value * x_std * y_std) / (x_var + y_var + (x_ave - y_ave) ** 2)
    return pcc_value, ccc_value


class RMSLELoss(Module):
    def __init__(self) -> None:
        """Initialize this object with the declared configuration.

        :rtype: None
        """
        super().__init__()
        self.mse = MSELoss()

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Forward.

        :param pred: Pred.
        :type pred: torch.Tensor
        :param actual: Actual.
        :type actual: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


def LDL_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ldl loss.

    :param output: Output.
    :type output: torch.Tensor
    :param target: Target.
    :type target: torch.Tensor
    :return: Result of this operation.
    :rtype: torch.Tensor
    """
    ldl = target.mul(torch.log(output))
    l1 = ldl.sum()
    l1 = l1 / target.shape[1]
    return -l1


def tensor_metrics(
    pred: torch.Tensor, true: torch.Tensor, label_scale: float = 0.1
) -> pd.DataFrame:
    """Tensor metrics.

    :param pred: Pred.
    :type pred: torch.Tensor
    :param true: True.
    :type true: torch.Tensor
    :param label_scale: Label scale.
    :type label_scale: float
    :return: Result of this operation.
    :rtype: pd.DataFrame
    """
    pred, true = pred / label_scale, true / label_scale
    device = true.device
    mse_loss = MSELoss().to(device)
    l1_loss = L1Loss().to(device)
    rmsle_loss = RMSLELoss().to(device)
    mae = l1_loss(pred, true)
    mse = mse_loss(pred, true)
    rmse = torch.sqrt(mse)
    r2 = r2_loss(pred, true)
    rmsle = rmsle_loss(pred, true)
    pcc, ccc = PCC(pred, true)
    # metrics = pd.DataFrame(data=[mae, mse, rmse, r2, rmsle], columns=["MAE", "MSE", "RMSE", "R2", "RMSLE"])
    metrics = {
        "MAE": [mae.item()],
        "MSE": [mse.item()],
        "RMSE": [rmse.item()],
        "R2": [r2.item()],
        "PCC": [pcc.item()],
        "CCC": [ccc.item()],
        "RMSLE": [rmsle.item()],
        "No.": [len(pred)],
    }
    metrics_df = pd.DataFrame.from_dict(metrics)
    # print(metrics_df)
    return metrics_df
