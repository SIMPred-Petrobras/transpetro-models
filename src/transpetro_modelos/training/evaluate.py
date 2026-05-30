import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from typing import Any

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

from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    train_df,
    contamination=0.05,
    n_estimators=100,
    random_state=42,
):
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    model.fit(train_df)

    return model

def compute_isolation_forest_errors(model, df: pd.DataFrame):
    """
    Retorna anomaly score positivo.
    Quanto MAIOR o valor, mais anômalo.
    """

    scores = -model.score_samples(df)

    return np.asarray(scores)


def score_isolation_forest_set(
    model,
    df: pd.DataFrame,
    threshold: float,
):
    scores = compute_isolation_forest_errors(model, df)

    result = pd.DataFrame(index=df.index)

    result["reconstruction_error"] = scores
    result["is_anomaly"] = scores > threshold

    return result

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

'''def failure_detection_metrics(
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
    }'''

def apply_debounce(scores: pd.DataFrame, consecutive: int = 1) -> pd.DataFrame:
    """
    Exige N pontos consecutivos anômalos antes de disparar um alarme.

    consecutive=1 equivale a nenhum debounce (comportamento original).
    Para dados em 5 min, consecutive=6 exige 30 min contínuos.
    Para dados horários, consecutive=6 exige 6 horas contínuas.

    O alarme é atribuído ao último ponto da sequência; pontos anteriores
    ao tamanho da janela ficam como False.
    """
    if consecutive <= 1:
        return scores
    rolling_count = scores["is_anomaly"].astype(int).rolling(consecutive, min_periods=consecutive).sum()
    debounced = (rolling_count >= consecutive).fillna(False)
    result = scores.copy()
    result["is_anomaly"] = debounced
    return result


def compute_balanced_score(
    scores_df: pd.DataFrame,
    failure_date: datetime,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
    *,
    false_positive_penalty: float = 2.0,
    min_prefailure_rate: float = 0.5,
    debounce_consecutive: int = 1,
) -> dict[str, Any]:
    """
    Calcula score balanceado que penaliza explicitamente falsos positivos.

    Lógica central:
        balanced_score = prefailure_alert_rate - (false_positive_penalty × normal_alert_rate)

    Isso garante que modelos com alta taxa de FP no período normal sejam
    rebaixados no ranking mesmo que também detectem bem o pré-falha.

    Args:
        scores_df: DataFrame com coluna 'is_anomaly' e índice temporal.
        failure_date: Data conhecida da falha do equipamento.
        prefailure_days: Tamanho da janela pré-falha em dias.
        normal_end_days: Quantos dias antes da falha o período "normal" termina.
            Ex: 60 → considera normal apenas amostras com >60 dias antes da falha.
        false_positive_penalty: Fator de penalização para alertas no período normal.
            Valor 2.0 significa que cada 1% de FP "custa" 2% no score final.
        min_prefailure_rate: Taxa mínima desejada de detecção pré-falha.
            Modelos abaixo desse patamar recebem penalização adicional.

    Returns:
        Dict com composite_score (0-1), balanced_score, discrimination_ratio,
        taxas de alerta por período e contagens absolutas.
    """

    if debounce_consecutive > 1:
        scores_df = apply_debounce(scores_df, consecutive=debounce_consecutive)

    # Definir janelas temporais
    prefailure_start = failure_date - pd.Timedelta(days=prefailure_days)
    normal_end = failure_date - pd.Timedelta(days=normal_end_days)

    # Período normal: amostras com mais de normal_end_days antes da falha
    normal_mask = scores_df.index < normal_end
    normal_samples = scores_df[normal_mask]
    n_normal = len(normal_samples)
    n_normal_alerts = int(normal_samples["is_anomaly"].sum())
    normal_alert_rate = float(normal_samples["is_anomaly"].mean()) if n_normal > 0 else 0.0

    # Período pré-falha: últimos prefailure_days antes da falha
    prefailure_mask = (scores_df.index >= prefailure_start) & (scores_df.index < failure_date)
    prefailure_samples = scores_df[prefailure_mask]
    n_prefailure = len(prefailure_samples)
    n_prefailure_alerts = int(prefailure_samples["is_anomaly"].sum())
    prefailure_alert_rate = float(prefailure_samples["is_anomaly"].mean()) if n_prefailure > 0 else 0.0

    # Discrimination ratio: quantas vezes mais alerta pré-falha vs normal
    if normal_alert_rate > 0:
        discrimination_ratio = prefailure_alert_rate / normal_alert_rate
    else:
        discrimination_ratio = float("inf") if prefailure_alert_rate > 0 else 1.0

    # Score balanceado: alta detecção pré-falha + baixos FP normais
    # Melhor caso: prefailure=1.0, normal=0.0 → balanced_score = 1.0
    # Pior caso:  prefailure=0.0, normal=0.5 → balanced_score = -1.0
    balanced_score = (prefailure_alert_rate - false_positive_penalty * (normal_alert_rate ** 2))

    # Penalização adicional se não atingir detecção mínima pré-falha
    if prefailure_alert_rate < min_prefailure_rate:
        penalty = (min_prefailure_rate - prefailure_alert_rate) * 2.0
        balanced_score -= penalty

    # Normalizar para [0, 1]
    composite_score = max(
        0.0,
        min(1.0, (balanced_score + false_positive_penalty) / (1.0 + false_positive_penalty))
    )

    return {
        "composite_score": composite_score,
        "balanced_score": balanced_score,
        "discrimination_ratio": discrimination_ratio,
        "prefailure_alert_rate": prefailure_alert_rate,
        "normal_alert_rate": normal_alert_rate,
        "n_prefailure_alerts": n_prefailure_alerts,
        "n_normal_alerts": n_normal_alerts,
        "n_prefailure_samples": n_prefailure,
        "n_normal_samples": n_normal,
        "false_positive_penalty_used": false_positive_penalty,
        "debounce_consecutive": debounce_consecutive,
    }


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
