"""
Detector de mudança no NÍVEL de um sensor, descontada a estação e a carga (modelo de comportamento normal).

O KS na série bruta compara um dia com a referência inteira (o ano todo). Em sensores com estação forte, como
temperatura de mancal, todo dia já parece "diferente" do ano: o limite vai para perto de 1 e um degrau no meio do
ano passa batido. No B-8802B, a temperatura do mancal LA caiu ~6 °C em 28/03/2026 e o KS bruto não disparou
(ficou rente ao limite de abril a julho).

Este detector:
1. ajusta, na referência, uma regressão linear do sensor-alvo contra sensores que carregam a estação e a carga
   (temperaturas do motor, pressões, corrente): o comportamento normal;
2. monitora o resíduo (medido − previsto), que já não tem estação, pela mediana diária;
3. dispara quando a mediana diária sai da faixa [q, 1−q] das medianas diárias da referência em k dos últimos n
   dias com operação (parada maior que `reset_gap` zera a contagem).

Validação (notebooks/drift/mudanca_conceito_B-8802B.ipynb):
- B-8802B, mancal LA, referência 2025: 0 disparos em 2025 e em jan–mar/2026; dispara em 01/04/2026, 4 dias depois
  do degrau (o KS bruto: nenhum disparo robusto).
- B-4064A, mancal LNA: volta do reparo (+20 °C) em 2 dias; 0 disparos falsos com referência de só 4 meses (o KS
  bruto dá 2, pela estação) e 0 com 12 meses; não dispara na rampa de 2 dias da falha (papel do modelo de anomalia).
Limite: precisa de uma regressão que explique o sensor (no mancal LNA do B-8802B, R² 0,32 e faixa de −6 a +22 °C:
não serve); e, como o KS, de referência longa o bastante para cobrir os regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ResidualLevelDetector:
    """Mediana diária do resíduo de um modelo de comportamento normal fora da faixa da referência."""

    def __init__(self, target: str, predictors: list[str], q: float = 0.005, k: int = 3, n: int = 5,
                 min_coverage: float = 0.5, reset_gap: str = "3D"):
        self.target, self.predictors = target, list(predictors)
        self.q, self.k, self.n = q, k, n
        self.min_coverage = min_coverage      # fração mínima do dia com operação para o dia contar
        self.reset_gap = pd.Timedelta(reset_gap)
        self.model: LinearRegression | None = None
        self.band: tuple[float, float] | None = None
        self.r2: float | None = None

    def residuals(self, df: pd.DataFrame) -> pd.Series:
        """Resíduo instante a instante (medido − previsto)."""
        return df[self.target] - self.model.predict(df[self.predictors])

    def daily(self, df: pd.DataFrame) -> pd.Series:
        """Mediana diária do resíduo, só nos dias com operação suficiente."""
        r = self.residuals(df)
        passo = df.index.to_series().diff().median()
        minimo = self.min_coverage * (pd.Timedelta("1D") / passo)
        n = r.resample("1D").count()
        return r.resample("1D").median()[n >= minimo]

    def fit(self, reference: pd.DataFrame) -> "ResidualLevelDetector":
        self.model = LinearRegression().fit(reference[self.predictors], reference[self.target])
        self.r2 = float(self.model.score(reference[self.predictors], reference[self.target]))
        lo, hi = self.daily(reference).quantile([self.q, 1 - self.q])
        self.band = (float(lo), float(hi))
        return self

    def scan(self, df: pd.DataFrame) -> tuple[pd.Series, list[pd.Timestamp]]:
        """(mediana diária do resíduo, disparos). Depois de cada disparo a contagem recomeça."""
        d = self.daily(df)
        lo, hi = self.band
        disparos, hist, anterior = [], [], None
        for t, v in d.items():
            if anterior is not None and t - anterior > self.reset_gap:
                hist = []
            anterior = t
            hist = (hist + [bool(v < lo or v > hi)])[-self.n:]
            if sum(hist) >= self.k:
                disparos.append(t)
                hist = []
        return d, disparos
