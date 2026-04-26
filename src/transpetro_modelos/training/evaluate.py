import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transpetro_modelos.training.train import make_windows


def compute_reconstruction_errors_sequence(
    model: torch.nn.Module,
    df: pd.DataFrame,
    seq_len: int,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Per-window MSE for sequence models (LSTM). Returns one error per window."""
    model.eval()
    windows = make_windows(df.values.astype("float32"), seq_len)
    tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstructed, _ = model(batch)
            mse = F.mse_loss(reconstructed, batch, reduction="none").mean(dim=[1, 2])
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def score_test_set_sequence(
    model: torch.nn.Module,
    df: pd.DataFrame,
    seq_len: int,
    threshold: float,
    batch_size: int = 512,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score a DataFrame using a sequence model. Error is assigned to the last timestamp of each window."""
    errors = compute_reconstruction_errors_sequence(model, df, seq_len, batch_size, device)
    timestamps = df.index[seq_len - 1 :]
    return pd.DataFrame(
        {"reconstruction_error": errors, "is_anomaly": errors > threshold},
        index=timestamps,
    )


def compute_reconstruction_errors(
    model: torch.nn.Module,
    df: pd.DataFrame,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Returns per-sample MSE reconstruction error."""
    model.eval()
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstructed, _ = model(batch)
            mse = F.mse_loss(reconstructed, batch, reduction="none").mean(dim=1)
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def determine_threshold(train_errors: np.ndarray, percentile: float = 95.0) -> float:
    """Threshold = percentile of training reconstruction errors."""
    return float(np.percentile(train_errors, percentile))


def score_test_set(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    threshold: float,
    batch_size: int = 512,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Compute reconstruction error and anomaly flag for test set.
    Returns DataFrame with original index + reconstruction_error + is_anomaly columns.
    """
    errors = compute_reconstruction_errors(model, test_df, batch_size=batch_size, device=device)
    return pd.DataFrame(
        {"reconstruction_error": errors, "is_anomaly": errors > threshold},
        index=test_df.index,
    )
