from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch.nn as nn
from dataset import *
from model import *
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str | torch.device,
) -> tuple[float, float, object, object]:
    """Evaluate model.

    :param model: Named model variant.
    :type model: torch.nn.Module
    :param test_loader: Test loader.
    :type test_loader: DataLoader
    :param criterion: Criterion.
    :type criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    :param device: Computation device accepted by the numerical backend.
    :type device: str | torch.device
    :return: Result of this operation.
    :rtype: tuple[float, float, object, object]
    """
    model.eval()
    total_loss = 0
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = (
                input_ids.to(device),
                graph_features.to(device),
                labels.to(device),
            )

            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy, all_preds, all_labels


def calculate_metrics(
    all_labels: Sequence[int], all_preds: Sequence[int]
) -> dict[str, float]:
    """Calculate metrics.

    :param all_labels: All labels.
    :type all_labels: Sequence[int]
    :param all_preds: All preds.
    :type all_preds: Sequence[int]
    :return: Result of this operation.
    :rtype: dict[str, float]
    """
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return acc, sen, spe, mcc, auc


def save_metrics_to_csv(
    metrics: Mapping[str, float], filename: str = "metrics.csv"
) -> None:
    """Save metrics to csv.

    :param metrics: Metrics.
    :type metrics: Mapping[str, float]
    :param filename: Filename.
    :type filename: str
    :rtype: None
    """
    df = pd.DataFrame(
        [metrics], columns=["ACC", "Sensitivity", "Specificity", "MCC", "AUC"]
    )
    df.to_csv(filename, index=False)


device = torch.device("mps" if torch.cuda.is_available() else "cpu")
vocab_size = len(Pep_residue2idx)
d_model, d_ff, n_layers, n_heads, max_len = 256, 512, 2, 4, 50
structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 50,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)],
}


test_sequences, test_graph_features, test_labels = load_data_from_fasta("test.txt")

test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)


model = ToxiPep_Model(
    vocab_size,
    d_model,
    d_ff,
    n_layers,
    n_heads,
    max_len,
    structural_config=structural_config,
).to(device)

model.load_state_dict(torch.load("best_model_0.9.pth"))
criterion = nn.CrossEntropyLoss()


_, _, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)


acc, sen, spe, mcc, auc = calculate_metrics(all_labels, all_preds)
metrics = {"ACC": acc, "Sensitivity": sen, "Specificity": spe, "MCC": mcc, "AUC": auc}

print(
    f"ACC: {acc:.4f}, Sensitivity: {sen:.4f}, Specificity: {spe:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}"
)
