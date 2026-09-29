from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


@dataclass
class DriftClassification:
    """
    Classificação causal-operacional de um evento de drift.

    regime:
        - "degradacao"
        - "mudanca_regime"
        - "indeterminado"
        - "sem_sinal"

    action:
        - "nao_retreinar"
        - "confirmar_operacao"
        - "continuar_monitorando"
    """

    regime: str
    action: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def should_retrain(self) -> bool:
        # Deliberadamente NÃO libera retreino automático.
        # O retrain_pipeline.py já possui o portão --operacao-confirmou.
        return self.action == "confirmar_operacao"


def _safe_median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def _daily_median(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.resample("D").median().dropna()


def _trend_metrics(
    series: pd.Series,
    *,
    days: int = 30,
    min_days: int = 15,
    slope_min: float = 0.05,
) -> dict[str, float | bool]:
    """
    Mede tendência robusta usando mediana diária + Theil-Sen.

    'consistencia' é a fração de diferenças diárias positivas.
    """
    daily = _daily_median(series).tail(days)

    if len(daily) < min_days:
        return {
            "slope_dia": np.nan,
            "consistencia": np.nan,
            "monotonicidade": False,
            "n_dias": float(len(daily)),
        }

    y = daily.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)

    slope, _, _, _ = theilslopes(y, x)

    diffs = np.diff(y)
    consistency_up = float(np.mean(diffs > 0)) if len(diffs) else np.nan

    # Mantém a ideia original do drift_report:
    # tendência suficientemente forte + persistente.
    monotonic = bool(
        np.isfinite(slope)
        and abs(slope) > slope_min
        and (consistency_up > 0.65 or consistency_up < 0.35)
    )

    return {
        "slope_dia": float(slope),
        "consistencia": consistency_up,
        "monotonicidade": monotonic,
        "n_dias": float(len(daily)),
    }


def _level_shift_metrics(
    error: pd.Series,
    event_time: pd.Timestamp,
    *,
    pre_days: int = 30,
    post_days: int = 14,
) -> dict[str, float | bool]:
    """
    Compara nível robusto do erro antes/depois do evento.

    Só utiliza dados <= event_time.
    """
    s = pd.to_numeric(error, errors="coerce").dropna()

    pre = s[
        (s.index < event_time)
        & (s.index >= event_time - pd.Timedelta(days=pre_days))
    ]

    post = s[
        (s.index >= event_time)
        & (s.index <= event_time + pd.Timedelta(days=post_days))
    ]

    # Em walk-forward, o futuro normalmente não existe ainda.
    # Portanto, se o post tiver poucos dados, usa somente o trecho
    # disponível até event_time.
    if len(post) < 5:
        post = s[
            (s.index <= event_time)
            & (s.index >= event_time - pd.Timedelta(days=post_days))
        ]

    pre_med = _safe_median(pre)
    post_med = _safe_median(post)

    if not np.isfinite(pre_med) or not np.isfinite(post_med):
        return {
            "pre_median": np.nan,
            "post_median": np.nan,
            "relative_change": np.nan,
            "level_shift": False,
        }

    relative_change = (
        (post_med - pre_med) / abs(pre_med)
        if abs(pre_med) > 1e-12
        else np.nan
    )

    return {
        "pre_median": pre_med,
        "post_median": post_med,
        "relative_change": float(relative_change),
        "level_shift": bool(
            np.isfinite(relative_change)
            and relative_change > 0.20
        ),
    }


def _plateau_metrics(
    error: pd.Series,
    event_time: pd.Timestamp,
    *,
    days: int = 14,
) -> dict[str, float | bool]:
    """
    Detecta se o erro parece ter entrado em um novo patamar estável.
    """
    s = pd.to_numeric(error, errors="coerce").dropna()

    post = s[
        (s.index >= event_time - pd.Timedelta(days=days))
        & (s.index <= event_time)
    ]

    if len(post) < 10:
        return {
            "plateau_cv": np.nan,
            "plateau_stable": False,
        }

    med = float(post.median())
    std = float(post.std())

    cv = std / abs(med) if abs(med) > 1e-12 else np.nan

    return {
        "plateau_cv": float(cv) if np.isfinite(cv) else np.nan,
        "plateau_stable": bool(
            np.isfinite(cv) and cv < 0.35
        ),
    }


def classify_drift_event(
    *,
    event_time: pd.Timestamp,
    reconstruction_error: pd.Series,
    physical_df: pd.DataFrame | None = None,
    physical_columns: dict[str, Iterable[str]] | None = None,
    drift_detected: bool = True,
    status: str | None = None,
) -> DriftClassification:
    """
    Classifica um evento de drift usando somente informação disponível
    até event_time.

    physical_columns:
        {
            "vibration": [...],
            "temperature": [...],
            "pressure": [...],
        }

    Se não informado, tenta descobrir vibração/temperatura pelos nomes.
    """

    if not drift_detected:
        return DriftClassification(
            regime="sem_sinal",
            action="continuar_monitorando",
            confidence=1.0,
            reasons=["Nenhum detector de drift disparou."],
        )

    event_time = pd.Timestamp(event_time)

    error = pd.to_numeric(
        reconstruction_error.loc[
            reconstruction_error.index <= event_time
        ],
        errors="coerce",
    ).dropna()

    if error.empty:
        return DriftClassification(
            regime="indeterminado",
            action="nao_retreinar",
            confidence=0.0,
            reasons=["Não há erro de reconstrução suficiente para classificar o evento."],
        )

    metrics: dict[str, float] = {}
    reasons: list[str] = []

    # ------------------------------------------------------------
    # 1. Tendência física
    # ------------------------------------------------------------

    physical_trends: dict[str, dict] = {}

    if physical_df is not None and not physical_df.empty:
        physical_df = physical_df.loc[
            physical_df.index <= event_time
        ].sort_index()

        if physical_columns is None:
            physical_columns = {
                "vibration": [
                    c for c in physical_df.columns
                    if "vibra" in str(c).lower()
                ],
                "temperature": [
                    c for c in physical_df.columns
                    if "temperatura" in str(c).lower()
                    or "temperature" in str(c).lower()
                ],
                "pressure": [
                    c for c in physical_df.columns
                    if "pressão" in str(c).lower()
                    or "pressao" in str(c).lower()
                    or "pressure" in str(c).lower()
                ],
            }

        for group in ("vibration", "temperature"):
            for column in physical_columns.get(group, []):
                if column not in physical_df.columns:
                    continue

                tm = _trend_metrics(physical_df[column])

                if np.isfinite(tm["slope_dia"]):
                    physical_trends[column] = tm

                    metrics[f"{column}__slope_dia"] = float(
                        tm["slope_dia"]
                    )
                    metrics[f"{column}__consistencia"] = float(
                        tm["consistencia"]
                    )

    degradation_signals = [
        (name, value)
        for name, value in physical_trends.items()
        if value["monotonicidade"]
        and value["slope_dia"] > 0
    ]

    # ------------------------------------------------------------
    # 2. Comportamento do erro
    # ------------------------------------------------------------

    level = _level_shift_metrics(error, event_time)
    plateau = _plateau_metrics(error, event_time)

    for key, value in {**level, **plateau}.items():
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            metrics[key] = float(value)

    # ------------------------------------------------------------
    # 3. Tendência do próprio erro
    # ------------------------------------------------------------

    error_trend = _trend_metrics(
        error,
        days=30,
        min_days=10,
        slope_min=0.0,
    )

    if np.isfinite(error_trend["slope_dia"]):
        metrics["error_slope_dia"] = float(error_trend["slope_dia"])
        metrics["error_consistencia"] = float(
            error_trend["consistencia"]
        )

    error_rising = bool(
        np.isfinite(error_trend["slope_dia"])
        and error_trend["slope_dia"] > 0
        and error_trend["consistencia"] > 0.60
    )

    # ------------------------------------------------------------
    # 4. Classificação
    # ------------------------------------------------------------

    # Caso 1: evidência física consistente de degradação.
    if degradation_signals:
        names = ", ".join(name for name, _ in degradation_signals)

        reasons.append(
            f"Tendência física monotônica de subida em: {names}."
        )

        if error_rising:
            reasons.append(
                "O erro de reconstrução também apresenta tendência de subida."
            )

        return DriftClassification(
            regime="degradacao",
            action="nao_retreinar",
            confidence=0.90 if error_rising else 0.80,
            reasons=reasons,
            metrics=metrics,
        )

    # Caso 2: mudança de patamar sem evidência física de degradação.
    if (
        level["level_shift"]
        and (
            plateau["plateau_stable"]
            or not error_rising
        )
    ):
        reasons.append(
            "O erro apresentou mudança persistente de nível sem "
            "tendência física monotônica de degradação."
        )

        if plateau["plateau_stable"]:
            reasons.append(
                "O erro parece estabilizar em um novo patamar."
            )

        return DriftClassification(
            regime="mudanca_regime",
            action="confirmar_operacao",
            confidence=0.80,
            reasons=reasons,
            metrics=metrics,
        )

    # Caso 3: detector disparou, mas as evidências são conflitantes.
    reasons.append(
        "O detector disparou, mas não há evidência suficiente para "
        "separar degradação de mudança de regime."
    )

    return DriftClassification(
        regime="indeterminado",
        action="nao_retreinar",
        confidence=0.40,
        reasons=reasons,
        metrics=metrics,
    )