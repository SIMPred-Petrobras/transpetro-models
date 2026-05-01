from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


@dataclass
class PreprocessingArtifacts:
    scaler: object | None = None
    clip_bounds: dict | None = None
    knn_imputer: KNNImputer | None = None


@dataclass
class PreprocessingReport:
    rows_before: int
    rows_after: int
    missing_before: int
    missing_after: int


def filter_running(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Remove rows where pump is considered off (column value below threshold).
    If column is not present (e.g. per-sensor mode), returns df unchanged."""
    if column not in df.columns:
        return df
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


def clip(
    df: pd.DataFrame,
    bounds=None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Clip values to [P_lower, P_upper] per column.
    If bounds is None, calculates from data (use on train set).
    Returns (clipped_df, bounds_dict).
    """
    if bounds is None:
        bounds = {}
        for col in df.columns:
            bounds[col] = (
                np.percentile(df[col].dropna(), lower_pct),
                np.percentile(df[col].dropna(), upper_pct),
            )

    df = df.copy()
    for col in df.columns:
        lo, hi = bounds[col]
        df[col] = df[col].clip(lo, hi)

    return df, bounds


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

def interpolate_df(df: pd.DataFrame, method="time", limit=3) -> pd.DataFrame:
    df = df.interpolate(method=method).bfill().ffill().dropna()
    return df

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


def moving_average(
    df: pd.DataFrame,
    window: int = 3,
    min_periods: int = 1,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply a causal rolling mean to selected columns."""
    if columns is None:
        columns = list(df.columns)

    df = df.copy()
    df[columns] = df[columns].rolling(window=window, min_periods=min_periods, center=False).mean()
    return df


def knn_impute(
    df: pd.DataFrame,
    imputer: KNNImputer | None = None,
    n_neighbors: int = 3,
    weights: str = "distance",
    metric: str = "nan_euclidean",
) -> tuple[pd.DataFrame, KNNImputer]:
    """Impute missing values using KNN, fitting only when imputer is None."""
    if imputer is None:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)
        values = imputer.fit_transform(df.values)
    else:
        values = imputer.transform(df.values)

    return pd.DataFrame(values, index=df.index, columns=df.columns), imputer


def run_preprocessing(
    df: pd.DataFrame,
    steps: list[dict],
    fitted_artifacts: PreprocessingArtifacts | None = None,
    fitted_scaler=None,
    fitted_clip_bounds=None,
    fitted_knn_imputer: KNNImputer | None = None,
    return_artifacts: bool = False,
    return_report: bool = False,
):
    """
    Execute a preprocessing pipeline defined as a list of step dicts.
    Returns (processed_df, scaler, clip_bounds).

    Steps example:
        [
            {"step": "filter_running", "column": "Corrente", "threshold": 1.0},
            {"step": "remove_transients", "minutes": 10},
            {"step": "clip"},
            {"step": "normalize", "method": "standard"},
        ]

    On the train set, pass fitted_scaler=None and fitted_clip_bounds=None.
    On val/test sets, pass the scaler and clip_bounds returned from the train call.
    """
    artifacts = fitted_artifacts or PreprocessingArtifacts(
        scaler=fitted_scaler,
        clip_bounds=fitted_clip_bounds,
        knn_imputer=fitted_knn_imputer,
    )
    report = PreprocessingReport(
        rows_before=len(df),
        rows_after=0,
        missing_before=int(df.isna().sum().sum()),
        missing_after=0,
    )

    for step_cfg in steps:
        step = step_cfg["step"]
        params = {k: v for k, v in step_cfg.items() if k != "step"}

        if step == "filter_running":
            df = filter_running(df, **params)
        elif step == "remove_transients":
            df = remove_transients(df, **params)
        elif step == "normalize":
            df, artifacts.scaler = normalize(df, scaler=artifacts.scaler, **params)
        elif step == "clip":
            df, artifacts.clip_bounds = clip(df, bounds=artifacts.clip_bounds, **params)
        elif step == "select_features":
            df = select_features(df, **params)
        elif step == "resample":
            df = resample(df, **params)
        elif step == "ffill":
            df = ffill(df, **params)
        elif step == "interpolate":
            df = interpolate_df(df, **params)
        elif step == "moving_average":
            df = moving_average(df, **params)
        elif step == "knn_impute":
            df, artifacts.knn_imputer = knn_impute(df, imputer=artifacts.knn_imputer, **params)
        elif step == "remove_sensor_errors":
            df = remove_sensor_errors(df, **params)
        else:
            raise ValueError(f"Unknown preprocessing step: '{step}'")

    report.rows_after = len(df)
    report.missing_after = int(df.isna().sum().sum())

    if return_artifacts and return_report:
        return df, artifacts, report
    if return_artifacts:
        return df, artifacts
    if return_report:
        return df, artifacts.scaler, artifacts.clip_bounds, report
    return df, artifacts.scaler, artifacts.clip_bounds
