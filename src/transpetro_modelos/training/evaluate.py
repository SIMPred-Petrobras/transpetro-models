from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transpetro_modelos.training.train import make_windows


def fit_ocsvm(train_df: pd.DataFrame, nu: float = 0.05, gamma: str = "scale"):
    from sklearn.svm import OneClassSVM

    clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    clf.fit(train_df.values.astype("float32"))
    return clf


def compute_ocsvm_errors(clf, df: pd.DataFrame) -> np.ndarray:
    return (-clf.decision_function(df.values.astype("float32"))).astype("float32")


def score_ocsvm_set(clf, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    errors = compute_ocsvm_errors(clf, df)
    return pd.DataFrame(
        {"reconstruction_error": errors, "is_anomaly": errors > threshold},
        index=df.index,
    )


def fit_isolation_forest(
    train_df: pd.DataFrame,
    n_estimators: int = 100,
    contamination: str = "auto",
    random_state: int = 42,
):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(train_df.values.astype("float32"))
    return clf


def compute_isolation_forest_errors(clf, df: pd.DataFrame) -> np.ndarray:
    # score_samples retorna valores maiores para normais; negamos para que maior = mais anômalo
    return (-clf.score_samples(df.values.astype("float32"))).astype("float32")


def score_isolation_forest_set(clf, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    errors = compute_isolation_forest_errors(clf, df)
    return pd.DataFrame(
        {"reconstruction_error": errors, "is_anomaly": errors > threshold},
        index=df.index,
    )


def fit_lof(
    train_df: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.05,
):
    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    clf.fit(train_df.values.astype("float32"))
    return clf


def compute_lof_errors(clf, df: pd.DataFrame) -> np.ndarray:
    # score_samples retorna valores negativos; negamos para que maior = mais anômalo
    return (-clf.score_samples(df.values.astype("float32"))).astype("float32")


def score_lof_set(clf, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    errors = compute_lof_errors(clf, df)
    return pd.DataFrame(
        {"reconstruction_error": errors, "is_anomaly": errors > threshold},
        index=df.index,
    )


def compute_vae_errors(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """MSE de reconstrução usando a média do espaço latente (sem sampling)."""
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            recon, _, _ = model(batch)
            mse = F.mse_loss(recon, batch, reduction="none").mean(dim=1)
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def score_vae_set(
    model: torch.nn.Module,
    df: pd.DataFrame,
    threshold: float,
    device: str = "cpu",
    batch_size: int = 512,
) -> pd.DataFrame:
    errors = compute_vae_errors(model, df, device=device, batch_size=batch_size)
    return pd.DataFrame(
        {"reconstruction_error": errors, "is_anomaly": errors > threshold},
        index=df.index,
    )


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


def failure_detection_metrics(
    scores: pd.DataFrame,
    failure_date: datetime,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
) -> dict[str, float | int]:
    """
    Métricas de detecção de falha a partir de um DataFrame de scores (coluna is_anomaly).

    Define dois períodos:
      - normal: tudo antes de (failure_date - normal_end_days)
      - pré-falha: (failure_date - prefailure_days) até failure_date

    Retorna:
      composite_score         = prefailure_alert_rate * (1 - normal_alert_rate)  [0..1, primário]
      discrimination_ratio    = prefailure_alert_rate / (normal_alert_rate + eps) [auxiliar]
      prefailure_alert_rate   = fração de alarmes na janela pré-falha
      normal_alert_rate       = fração de alarmes no período normal
    """
    _EPS = 1e-9
    failure_ts = pd.Timestamp(failure_date)
    normal_end = failure_ts - pd.Timedelta(days=normal_end_days)
    prefailure_start = failure_ts - pd.Timedelta(days=prefailure_days)

    normal_flags = scores.loc[scores.index < normal_end, "is_anomaly"]
    prefailure_flags = scores.loc[
        (scores.index >= prefailure_start) & (scores.index < failure_ts), "is_anomaly"
    ]

    normal_rate = float(normal_flags.mean()) if len(normal_flags) > 0 else 0.0
    prefailure_rate = float(prefailure_flags.mean()) if len(prefailure_flags) > 0 else 0.0

    return {
        "composite_score": prefailure_rate * (1.0 - normal_rate),
        "discrimination_ratio": prefailure_rate / (normal_rate + _EPS),
        "prefailure_alert_rate": prefailure_rate,
        "normal_alert_rate": normal_rate,
        "n_prefailure_alerts": int(prefailure_flags.sum()),
        "n_normal_alerts": int(normal_flags.sum()),
        "n_prefailure_samples": len(prefailure_flags),
        "n_normal_samples": len(normal_flags),
    }


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
