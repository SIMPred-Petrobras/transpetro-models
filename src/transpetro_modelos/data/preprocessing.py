import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional


def filter_running(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Remove rows where pump is considered off (column value below threshold)."""
    return df[df[column] > threshold].copy()


def remove_transients(df: pd.DataFrame, minutes: int = 10) -> pd.DataFrame:
    """
    Remove the first N minutes after each pump restart.
    Detects restarts as gaps > 5 minutes in the index.
    """
    if len(df) == 0:
        return df

    time_diff = df.index.to_series().diff()
    gap_threshold = pd.Timedelta(minutes=5)
    restart_mask = time_diff > gap_threshold
    restart_indices = df.index[restart_mask]

    mask = pd.Series(True, index=df.index)
    # Always remove first N minutes of the dataset
    if len(df) > 0:
        cutoff = df.index[0] + pd.Timedelta(minutes=minutes)
        mask[df.index < cutoff] = False

    for restart_time in restart_indices:
        cutoff = restart_time + pd.Timedelta(minutes=minutes)
        mask[(df.index >= restart_time) & (df.index < cutoff)] = False

    return df[mask].copy()


def normalize(
    df: pd.DataFrame,
    method: str = "standard",
    scaler=None,
) -> tuple[pd.DataFrame, object]:
    """
    Normalize features. If scaler is None, fits a new one (use on train set).
    Returns (normalized_df, fitted_scaler).
    """
    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    if method not in scalers:
        raise ValueError(f"method must be one of {list(scalers.keys())}, got '{method}'")

    if scaler is None:
        scaler = scalers[method]()
        values = scaler.fit_transform(df.values)
    else:
        values = scaler.transform(df.values)

    return pd.DataFrame(values, index=df.index, columns=df.columns), scaler


def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select a subset of columns."""
    return df[features].copy()


def remove_sensor_errors(df: pd.DataFrame, error_values: list[float] | None = None) -> pd.DataFrame:
    """Replace known sensor error codes with NaN (e.g., -25.0 in temperature sensors)."""
    if error_values is None:
        error_values = [-25.0]
    df = df.copy()
    for val in error_values:
        df = df.replace(val, np.nan)
    return df


def resample(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Resample COV (change-on-value) time series to a regular grid.
    Uses .last() to preserve the last recorded value in each window.
    NaN filling is intentionally NOT done here — use the ffill step after split
    to avoid data leakage between train and test sets.
    """
    return df.resample(freq).last()


def ffill(df: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    """
    Forward-fill NaN values up to `limit` consecutive periods.
    Rows that remain NaN after fill (gaps longer than limit) are dropped.
    Apply this step AFTER the train/test split to avoid data leakage.
    """
    return df.ffill(limit=limit).dropna()


def run_preprocessing(
    df: pd.DataFrame,
    steps: list[dict],
    fitted_scaler=None,
) -> tuple[pd.DataFrame, object]:
    """
    Execute a preprocessing pipeline defined as a list of step dicts.
    Returns (processed_df, scaler) — scaler is None if no normalize step.

    Steps example:
        [
            {"step": "filter_running", "column": "Corrente", "threshold": 1.0},
            {"step": "remove_transients", "minutes": 10},
            {"step": "normalize", "method": "standard"},
        ]

    On the train set, pass fitted_scaler=None (fits a new scaler).
    On val/test sets, pass the scaler returned from the train call.
    """
    scaler = fitted_scaler

    for step_cfg in steps:
        step = step_cfg["step"]
        params = {k: v for k, v in step_cfg.items() if k != "step"}

        if step == "filter_running":
            df = filter_running(df, **params)
        elif step == "remove_transients":
            df = remove_transients(df, **params)
        elif step == "normalize":
            df, scaler = normalize(df, scaler=scaler, **params)
        elif step == "select_features":
            df = select_features(df, **params)
        elif step == "resample":
            df = resample(df, **params)
        elif step == "ffill":
            df = ffill(df, **params)
        elif step == "remove_sensor_errors":
            df = remove_sensor_errors(df, **params)
        else:
            raise ValueError(f"Unknown preprocessing step: '{step}'")

    return df, scaler
