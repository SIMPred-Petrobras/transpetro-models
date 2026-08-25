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
    load_residual_coefs: dict | None = None  # {col: (intercept, slope)} de Temp~carga, ajustado no train


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

def remove_negatives(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df.loc[df[column] < 1, column] = np.nan
    return df

def filter_threshold(df: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    """Keep rows where ALL given columns exceed threshold (branch Lara).
    Missing columns are ignored; if none exist, returns df unchanged."""
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df
    mask = df[existing].gt(threshold).all(axis=1)
    return df[mask].copy()

def remove_transients(df: pd.DataFrame, minutes: int = 10, gap_minutes: int = 5) -> pd.DataFrame:
    """
    Remove the first N minutes after each pump restart.
    Detects restarts as gaps > gap_minutes in the index (default 5 min).
    For hourly resampled data, use gap_minutes=90 to avoid treating
    every 1-h step as a restart.
    """
    if len(df) == 0:
        return df

    time_diff = df.index.to_series().diff()
    gap_threshold = pd.Timedelta(minutes=gap_minutes)
    restart_mask = time_diff > gap_threshold
    restart_indices = df.index[restart_mask]

    mask = pd.Series(True, index=df.index)
    if len(df) > 0:
        cutoff = df.index[0] + pd.Timedelta(minutes=minutes)
        mask[df.index < cutoff] = False

    for restart_time in restart_indices:
        cutoff = restart_time + pd.Timedelta(minutes=minutes)
        mask[(df.index >= restart_time) & (df.index < cutoff)] = False

    return df[mask].copy()


def remove_regime_transients(
    df: pd.DataFrame,
    columns: list[str],
    deltas: list[float],
    minutes: int = 90,
    window: int = 3,
) -> pd.DataFrame:
    """
    Remove os `minutes` seguintes a um DEGRAU brusco de processo (ex.: manobra de pressão).

    Um degrau é |x_t - x_{t-window}| > delta em qualquer coluna listada (`window` em linhas da
    grade já reamostrada; ex.: 3 linhas de 5 min = 15 min). Mesma ideia do remove_transients
    (que trata partidas), mas disparada por mudança de regime operacional — evita que o
    autoencoder acuse manobras de processo como anomalia do equipamento. Colunas ausentes são
    ignoradas. Use APÓS resample/ffill e ANTES de select_features.
    """
    if len(df) == 0:
        return df
    step = pd.Series(False, index=df.index)
    for col, delta in zip(columns, deltas):
        if col in df.columns:
            step |= df[col].diff(window).abs() > delta
    if not step.any():
        return df
    last_step = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    last_step[step] = df.index[step]
    last_step = last_step.ffill()
    since = (df.index - last_step)
    in_mask = (since >= pd.Timedelta(0)) & (since < pd.Timedelta(minutes=minutes))
    return df[~in_mask.fillna(False).values].copy()

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
    df = df.interpolate(method=method, limit=limit)
    return df

def remove_sensor_errors(df: pd.DataFrame, error_values: list[float] | None = None) -> pd.DataFrame:
    """Replace known sensor error codes with NaN (e.g., -25.0 in temperature sensors)."""
    if error_values is None:
        error_values = [-25.0]
    df = df.copy()
    for val in error_values:
        df = df.replace(val, np.nan)
    return df


def resample(
    df: pd.DataFrame,
    freq: str = "1h",
    agg: str = "last",
    extra_aggs: dict | None = None,
) -> pd.DataFrame:
    """
    Resample COV (change-on-value) time series to a regular grid.

    agg : agregação base por coluna. 'last' (default) preserva o comportamento original
        (último valor reportado na janela), adequado para série COV.
    extra_aggs : opcional, {coluna: [aggs]} — ADICIONA colunas `{coluna}__{agg}` computadas
        sobre a janela de resample, sem remover as originais. Útil para recuperar o PICO de
        vibração que o 'last' descarta. aggs suportados: 'max', 'min', 'mean', 'std', 'rms'.
        Default None → comportamento original 100% inalterado (mudança opt-in).

    NaN filling is intentionally NOT done here — use the ffill step after split
    to avoid data leakage between train and test sets.
    """
    r = df.resample(freq)
    base = r.last() if agg == "last" else getattr(r, agg)()
    if not extra_aggs:
        return base
    out = base.copy()
    for col, aggs in extra_aggs.items():
        if col not in df.columns:
            continue
        for a in aggs:
            if a == "rms":
                out[f"{col}__rms"] = np.sqrt((df[col] ** 2).resample(freq).mean())
            else:
                out[f"{col}__{a}"] = getattr(df[col].resample(freq), a)()
    return out


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


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    include_diff: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Adiciona features de janela deslizante causal (média, desvio padrão, diff) lado a lado
    com as colunas originais.

    Parâmetros
    ----------
    windows : list[int]
        Tamanhos das janelas em número de períodos. Ex.: [6, 24] para dados horários
        equivale a 6 h e 24 h. Default: [6, 24].
    include_diff : bool
        Se True, inclui a primeira diferença (taxa de mudança) de cada sensor.
    columns : list[str] | None
        Colunas a expandir. Default: todas.

    Notas
    -----
    - Usa rolling causal (center=False), sem look-ahead.
    - Linhas com NaN gerados no início de cada janela são removidas via dropna().
    - Aplique este passo ANTES do clip/normalize no preset para que os limites
      de clipping sejam calculados sobre as features enriquecidas.
    - Em val/test o mesmo conjunto de colunas que o train produz é gerado, pois
      a lista de colunas é derivada dos dados de entrada.
    """
    if windows is None:
        windows = [6, 24]
    if columns is None:
        columns = list(df.columns)

    df = df.copy()
    for col in columns:
        for w in windows:
            min_p = max(2, w // 2)
            df[f"{col}__std{w}"] = df[col].rolling(w, min_periods=min_p).std()
            df[f"{col}__mean{w}"] = df[col].rolling(w, min_periods=min_p).mean()
        if include_diff:
            df[f"{col}__diff"] = df[col].diff()

    return df.dropna()


def add_load_residual(
    df: pd.DataFrame,
    temp_columns: list[str],
    load_column: str = "Corrente",
    replace: bool = False,
    coefs: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Adiciona (ou substitui) features de RESÍDUO de temperatura condicionado à carga:
    resid = Temp - (a + b * carga), com (a, b) de uma regressão linear simples Temp~carga.

    Captura "mancal mais quente do que a carga explica" — a assinatura de roçamento/desgaste —
    sendo INVARIANTE a trocas de regime de carga (que mudam Temp e carga juntas). Útil quando o
    nível absoluto de temperatura muda entre regimes (ex.: B-4064A pós-reparo).

    Ajuste (coefs=None) deve ser feito no TRAIN; em val/test passe os `coefs` retornados para
    evitar vazamento. Se replace=True, substitui a coluna de temperatura pelo resíduo; senão
    adiciona uma coluna `{col}__resid`.
    """
    df = df.copy()
    fitted = coefs is not None
    coefs = dict(coefs) if coefs else {}
    x = df[load_column].values.astype("float64")
    for col in temp_columns:
        if not fitted:
            mask = (~np.isnan(x)) & (~np.isnan(df[col].values))
            if mask.sum() >= 2:
                b, a = np.polyfit(x[mask], df[col].values[mask], 1)  # slope, intercept
            else:
                b, a = 0.0, float(np.nanmean(df[col].values))
            coefs[col] = (float(a), float(b))
        a, b = coefs[col]
        resid = df[col].values - (a + b * x)
        if replace:
            df[col] = resid
        else:
            df[f"{col}__resid"] = resid
    return df, coefs


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
        elif step == "remove_negatives":
            df = remove_negatives(df, **params)
        elif step == "filter_threshold":
            df = filter_threshold(df, **params)
        elif step == "remove_transients":
            df = remove_transients(df, **params)
        elif step == "remove_regime_transients":
            df = remove_regime_transients(df, **params)
        elif step == "normalize":
            df, artifacts.scaler = normalize(df, scaler=artifacts.scaler, **params)
        elif step == "clip":
            df, artifacts.clip_bounds = clip(df, bounds=artifacts.clip_bounds, **params)
        elif step == "select_features":
            df = select_features(df, **params)
        elif step == "remove_sensor_errors":
            df = remove_sensor_errors(df, **params)
        elif step == "interpolate":
            df = interpolate_df(df, **params)
        elif step == "resample":
            df = resample(df, **params)
        elif step == "ffill":
            df = ffill(df, **params)
        elif step == "moving_average":
            df = moving_average(df, **params)
        elif step == "knn_impute":
            df, artifacts.knn_imputer = knn_impute(df, imputer=artifacts.knn_imputer, **params)
        elif step == "add_rolling_features":
            df = add_rolling_features(df, **params)
        elif step == "add_load_residual":
            df, artifacts.load_residual_coefs = add_load_residual(
                df, coefs=artifacts.load_residual_coefs, **params
            )
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
