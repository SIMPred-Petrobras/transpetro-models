from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from skimage.filters import threshold_otsu
from itertools import combinations_with_replacement

@dataclass
class ResidualModel:
    targets: list
    regressors: list
    windows_h: list
    degree: int
    suffix: str
    drop_raw: bool
    mu: np.ndarray
    sd: np.ndarray
    lo: np.ndarray | None
    hi: np.ndarray | None
    coef: np.ndarray
    drop_inputs: list | None = None

@dataclass
class PreprocessingArtifacts:
    scaler: object | None = None
    clip_bounds: dict | None = None
    knn_imputer: KNNImputer | None = None
    residual_model: ResidualModel | None = None


@dataclass
class PreprocessingReport:
    rows_before: int
    rows_after: int
    missing_before: int
    missing_after: int

def _residual_base_features(df, regressors, windows_h):
    cols = {}
    for r in regressors:
        x = df[r].astype(float)
        cols[r] = x
        for w in windows_h:  # média móvel causal por TEMPO (robusta a buracos do filter_running)
            cols[f"{r}__mean{w:g}h"] = x.rolling(f"{int(round(w * 60))}min", min_periods=1).mean()
    return pd.DataFrame(cols, index=df.index)


def _expand_poly(Z, degree):
    if degree == 1:
        return Z
    if degree == 2:
        extra = [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(Z.shape[1]), 2)]
        return np.column_stack([Z] + extra)
    raise ValueError("degree deve ser 1 ou 2")


def fit_residual_model(df, targets, regressors, windows_h=(3,), degree=1, alpha=1e-3, suffix="__resid", drop_raw=True, clip_regressors=None, min_rows=48, drop_inputs=None):
    targets, regressors = list(targets), list(regressors)
    faltam = [c for c in targets + regressors if c not in df.columns]
    if faltam:
        raise ValueError(f"residualize: colunas ausentes: {faltam}")
    if set(targets) & set(regressors):
        raise ValueError("residualize: um alvo não pode ser regressor de si mesmo.")
    X = _residual_base_features(df, regressors, windows_h)
    Y = df[targets].astype(float)
    ok = (X.notna().all(axis=1) & Y.notna().all(axis=1)).to_numpy()
    if ok.sum() < min_rows:
        raise ValueError(f"residualize: só {int(ok.sum())} linhas válidas para ajustar (mín {min_rows}).")
    Xv, Yv = X.to_numpy()[ok], Y.to_numpy()[ok]
    if clip_regressors is None:
        clip_regressors = degree > 1
    lo, hi = ((np.percentile(Xv, 0.5, axis=0), np.percentile(Xv, 99.5, axis=0))
              if clip_regressors else (None, None))
    if clip_regressors:
        Xv = np.clip(Xv, lo, hi)
    mu, sd = Xv.mean(axis=0), Xv.std(axis=0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(Xv)), _expand_poly((Xv - mu) / sd, degree)])
    reg = alpha * len(A) * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ Yv)
    return ResidualModel(targets, regressors, list(windows_h), degree, suffix, drop_raw,
                         mu, sd, lo, hi, coef, drop_inputs=list(drop_inputs or []))

def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove colunas (ignora as que não existem)."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def add_pair_deltas(df, pairs, mode="abs", drop_raw=True, min_ref_abs=1.0):
    """Diferença entre sensores de cada par: (a, b) -> a - b. Sem parâmetros ajustados."""
    cols = [c for p in pairs for c in p]
    faltam = [c for c in cols if c not in df.columns]
    if faltam:
        raise ValueError(f"add_pair_deltas: colunas ausentes: {faltam}")
    if mode not in ("abs", "pct", "both"):
        raise ValueError("mode deve ser 'abs', 'pct' ou 'both'")
    out = df.copy()
    for a, b in pairs:
        d = df[a].astype(float) - df[b].astype(float)
        base = f"{a}__menos__{b}"
        if mode in ("abs", "both"):
            out[f"{base}__abs"] = d
        if mode in ("pct", "both"):
            ref = df[b].astype(float).where(df[b].abs() >= min_ref_abs)
            out[f"{base}__pct"] = 100.0 * d / ref.abs()
    if drop_raw:
        out = out.drop(columns=list(dict.fromkeys(cols)))
    return out.dropna()

def apply_residual_model(df, m: ResidualModel):
    faltam = [c for c in m.targets + m.regressors if c not in df.columns]
    if faltam:
        raise ValueError(f"residualize: colunas ausentes: {faltam}")
    Xv = _residual_base_features(df, m.regressors, m.windows_h).to_numpy()
    if m.lo is not None:
        Xv = np.clip(Xv, m.lo, m.hi)
    A = np.column_stack([np.ones(len(Xv)), _expand_poly((Xv - m.mu) / m.sd, m.degree)])
    pred = A @ m.coef
    out = df.copy()
    novas = []
    for j, t in enumerate(m.targets):
        out[t + m.suffix] = df[t].astype(float).to_numpy() - pred[:, j]
        novas.append(t + m.suffix)
    if m.drop_raw:
        out = out.drop(columns=m.targets)
    
    sobras = [c for c in (m.drop_inputs or []) if c in out.columns]
    if sobras:
        out = out.drop(columns=sobras)
    return out.dropna(subset=novas)

def filter_running(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Remove rows where pump is considered off (column value below threshold).
    If column is not present (e.g. per-sensor mode), returns df unchanged."""
    if column not in df.columns:
        return df
    return df[df[column] > threshold].copy()

from skimage.filters import threshold_otsu
import pandas as pd

def filter_threshold(
    df: pd.DataFrame,
    columns: list[str],
    threshold,
    mode: str = "all",
    fixed_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:

    existing = [c for c in columns if c in df.columns]

    if not existing:
        return df

    # -------------------------
    # CASO 1: threshold fixo (float)
    # -------------------------
    if isinstance(threshold, (int, float)):
        flags = df[existing].gt(threshold)

    # -------------------------
    # CASO 2: threshold por coluna (dict)
    # -------------------------
    elif isinstance(threshold, dict):
        flags = pd.DataFrame({
            col: df[col] > threshold[col]
            for col in existing
            if col in threshold
        })

    # -------------------------
    # CASO 3: OTSU automático
    # -------------------------
    elif threshold == "otsu":
        th_dict = {
            col: threshold_otsu(df[col].dropna().to_numpy())
            for col in existing
        }

        # Sobrescreve os thresholds das variáveis desejadas
        if fixed_thresholds is not None:
            th_dict.update(fixed_thresholds)

        flags = pd.DataFrame({
            col: df[col] > th_dict[col]
            for col in existing
        })

    else:
        raise ValueError("threshold must be float, dict or 'otsu'")

    if mode == "any":
        mask = flags.any(axis=1)
    elif mode == "all":
        mask = flags.all(axis=1)
    else:
        raise ValueError("mode must be 'any' or 'all'")

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
    df = df.interpolate(method=method, limit=limit).bfill().ffill().dropna()
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
        elif step == "filter_threshold":
            df = filter_threshold(df, **params)
        elif step == "remove_transients":
            df = remove_transients(df, **params)
        elif step == "normalize":
            df, artifacts.scaler = normalize(df, scaler=artifacts.scaler, **params)
        elif step == "clip":
            df, artifacts.clip_bounds = clip(df, bounds=artifacts.clip_bounds, **params)
        elif step == "remove_regime_transients":
            df = remove_regime_transients(df, **params)
        elif step == "add_rolling_features":
            df = add_rolling_features(df, **params)
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
        elif step == "residualize":
            if artifacts.residual_model is None:         
                artifacts.residual_model = fit_residual_model(df, **params)
            df = apply_residual_model(df, artifacts.residual_model)
        elif step == "drop_columns":
            df = drop_columns(df, **params)
        elif step == "add_pair_deltas":
            df = add_pair_deltas(df, **params)
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
