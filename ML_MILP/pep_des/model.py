"""
Author: Marco Bühler
"""

import torch, copy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    """Basic 2 Layer NN"""

    def __init__(self, inp, f1, f2, output):
        super().__init__()
        self.input = nn.Linear(inp, f1)
        self.f1 = nn.Linear(f1, f2)
        self.f2 = nn.Linear(f2, output)

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = self.f1(x)
        x = nn.functional.relu(x)
        x = self.f2(x)
        return x


def _create_loaders(x, y):
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.1, random_state=123, shuffle=True)

    training_generator = DataLoader(
        TensorDataset(
            torch.tensor(x_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )
    validation_generator = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=x_val.shape[0],
    )

    return training_generator, validation_generator


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    epochs: int = 1000,
    optimizer: torch.optim.Optimizer = None,
    verbose: bool = False,
    device: str = "cpu",
    seq_model: bool = False,
    loss=nn.MSELoss(),
) -> tuple[torch.nn.Module, list, list]:
    """Main Training Procedure"""
    training_generator, validation_generator = _create_loaders(x, y)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 400, 600, 800], gamma=0.9
    )  # Somewhat works ok
    model.to(device)

    loss_function = loss.to(device)
    train_loss = []
    val_loss = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for local_batch, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            yhat = model(local_batch)
            optimizer.zero_grad(set_to_none=True)
            if seq_model:
                loss = loss_function(yhat, local_labels)
            else:
                loss = loss_function(yhat, local_labels.reshape(-1, 1))
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())

        # Validation Set
        model.eval()
        with torch.no_grad():
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                yhat = model(local_batch)
                if seq_model:
                    loss = loss_function(yhat, local_labels)
                else:
                    loss = loss_function(yhat, local_labels.reshape(-1, 1))
            val_loss.append(loss.item())

            if val_loss[-1] < best_val_loss:
                best_mod = copy.deepcopy(model)
                best_val_loss = loss

        if epoch % 100 == 0 and verbose:
            print("Epoch %d train loss: %.6f, validation loss: %.6f" % (epoch + 1, train_loss[-1], val_loss[-1]))
    return best_mod, train_loss, val_loss


