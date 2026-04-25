import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


def make_dataloader(
    df,
    batch_size: int,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_sequence_dataloader(
    df,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    data = df.values.astype("float32")
    windows = np.stack([data[i : i + seq_len] for i in range(len(data) - seq_len + 1)])
    tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    logger=None,  # clearml Logger
) -> torch.nn.Module:
    """
    Train the autoencoder with early stopping.
    Logs train_loss and val_loss per epoch via ClearML logger if provided.
    Returns the model with best validation loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = F.mse_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                reconstructed, _ = model(batch)
                loss = F.mse_loss(reconstructed, batch)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        if logger is not None:
            logger.report_scalar("loss", "train", avg_train, epoch)
            logger.report_scalar("loss", "validation", avg_val, epoch)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.6f})")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} — train: {avg_train:.6f}, val: {avg_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
