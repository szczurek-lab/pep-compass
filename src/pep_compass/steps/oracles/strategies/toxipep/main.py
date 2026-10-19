from __future__ import annotations

from typing import TYPE_CHECKING

from dataset import *
from model import *

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> float:
    """Train model.

    :param model: Named model variant.
    :type model: torch.nn.Module
    :param train_loader: Train loader.
    :type train_loader: DataLoader
    :param criterion: Criterion.
    :type criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    :param optimizer: Optimizer.
    :type optimizer: torch.optim.Optimizer
    :param device: Computation device accepted by the numerical backend.
    :type device: str | torch.device
    :return: Result of this operation.
    :rtype: float
    """
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, graph_features, labels = batch
        input_ids, graph_features, labels = (
            input_ids.to(device),
            graph_features.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, graph_features, device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str | torch.device,
) -> tuple[float, float]:
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
    :rtype: tuple[float, float]
    """
    model.eval()
    total_loss = 0
    correct = 0
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
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy


vocab_size = len(Pep_residue2idx)
d_model = 256
d_ff = 512
n_layers = 2
n_heads = 4
max_len = 50


structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 50,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)],
}

train_sequences, train_graph_features, train_labels = load_data_from_fasta("train.txt")
test_sequences, test_graph_features, test_labels = load_data_from_fasta("test.txt")


train_dataset = MyDataSet(train_sequences, train_graph_features, train_labels)
test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)


train_loader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxiPep_Model(
    vocab_size,
    d_model,
    d_ff,
    n_layers,
    n_heads,
    max_len,
    structural_config=structural_config,
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

n_epochs = 100
best_accuracy = 0.0
best_model_path = "best_model_0.9.pth"

for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"新最佳模型保存于 Epoch {epoch + 1}，准确度: {best_accuracy:.4f}")
