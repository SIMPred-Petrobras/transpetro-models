"""
drift_detectors.py
==================================
Detectores de concept drift para série temporal de scores/erros de reconstrução
(ou qualquer feature contínua). Cada detector implementa uma interface comum
online (`update(x) -> bool`) para poder ser testado ponto-a-ponto, como um
stream real seria consumido em produção.

Detectores implementados:
    - KSDriftDetector        : Kolmogorov-Smirnov (janela de referência vs janela atual)
    - PSIDriftDetector       : Population Stability Index (bucketizado)
    - PageHinkleyDetector    : detecção sequencial de mudança de média (online, O(1))
    - CUSUMDriftDetector     : soma cumulativa (online, O(1))
    - ADWINLiteDetector      : janela adaptativa simplificada (sem dependência externa)

Todos herdam de `BaseDriftDetector`, então dá pra adicionar novos detectores
sem mexer no benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ════════════════════════════════════════════════════════════════
# Interface comum
# ════════════════════════════════════════════════════════════════

class BaseDriftDetector(ABC):
    """Interface comum: consome um valor por vez, retorna True quando dispara alarme.

    Depois de disparar, o detector fica "em alarme" até `reset()` ser chamado
    explicitamente pelo harness de benchmark (isso evita contar múltiplos
    alarmes redundantes para o mesmo evento de drift).
    """

    name: str = "base"

    def __init__(self) -> None:
        self._in_alarm = False

    @abstractmethod
    def _update(self, x: float) -> bool:
        """Lógica específica do detector. Retorna True se disparou agora."""

    def update(self, x: float) -> bool:
        if self._in_alarm:
            return False  # já alarmado, ignora até reset()
        fired = self._update(float(x))
        if fired:
            self._in_alarm = True
        return fired

    @abstractmethod
    def reset(self) -> None:
        """Reseta estado interno (referência, estatísticas acumuladas etc.)."""
        self._in_alarm = False


# ════════════════════════════════════════════════════════════════
# Kolmogorov-Smirnov
# ════════════════════════════════════════════════════════════════

class KSDriftDetector(BaseDriftDetector):
    """Compara janela de referência (dados "normais" conhecidos) contra uma
    janela deslizante do stream atual via teste KS de duas amostras.

    Dispara quando p-value < alpha, ou seja, quando as duas distribuições
    deixam de ser estatisticamente compatíveis.
    """

    name = "ks_test"

    def __init__(
        self,
        reference: np.ndarray,
        window_size: int = 50,
        alpha: float = 0.01,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.reference = np.asarray(reference, dtype=float)
        self.window_size = window_size
        self.alpha = alpha
        self.stride = stride
        self._buffer: list[float] = []
        self._since_last_test = 0

    def _update(self, x: float) -> bool:
        self._buffer.append(x)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        self._since_last_test += 1
        if len(self._buffer) < self.window_size or self._since_last_test < self.stride:
            return False
        self._since_last_test = 0

        _, p_value = stats.ks_2samp(self.reference, np.array(self._buffer))
        return bool(p_value < self.alpha)

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._since_last_test = 0


# ════════════════════════════════════════════════════════════════
# PSI (Population Stability Index)
# ════════════════════════════════════════════════════════════════

class PSIDriftDetector(BaseDriftDetector):
    """PSI clássico de crédito/risco, adaptado para stream: bucketiza a
    referência em quantis fixos e compara a distribuição da janela atual
    contra essa bucketização. Regra de bolso: PSI > 0.2 = drift relevante.
    """

    name = "psi"

    def __init__(
        self,
        reference: np.ndarray,
        window_size: int = 50,
        n_buckets: int = 10,
        threshold: float = 0.2,
        stride: int = 1,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        reference = np.asarray(reference, dtype=float)
        quantiles = np.linspace(0, 1, n_buckets + 1)
        self.edges = np.unique(np.quantile(reference, quantiles))
        if len(self.edges) < 3:
            # feature quase constante: força bins artificiais para evitar divisão por zero
            self.edges = np.linspace(reference.min() - eps, reference.max() + eps, n_buckets + 1)
        self.ref_counts, _ = np.histogram(reference, bins=self.edges)
        self.ref_frac = self.ref_counts / max(self.ref_counts.sum(), 1)
        self.window_size = window_size
        self.threshold = threshold
        self.stride = stride
        self.eps = eps
        self._buffer: list[float] = []
        self._since_last_test = 0

    def _update(self, x: float) -> bool:
        self._buffer.append(x)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        self._since_last_test += 1
        if len(self._buffer) < self.window_size or self._since_last_test < self.stride:
            return False
        self._since_last_test = 0

        cur_counts, _ = np.histogram(self._buffer, bins=self.edges)
        cur_frac = cur_counts / max(cur_counts.sum(), 1)

        ref = np.clip(self.ref_frac, self.eps, None)
        cur = np.clip(cur_frac, self.eps, None)
        psi = float(np.sum((cur - ref) * np.log(cur / ref)))
        return psi > self.threshold

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._since_last_test = 0


# ════════════════════════════════════════════════════════════════
# Page-Hinkley (sequencial, O(1) por ponto — bom pra produção em tempo real)
# ════════════════════════════════════════════════════════════════

class PageHinkleyDetector(BaseDriftDetector):
    """Detecta mudança de média de forma sequencial. Muito mais barato que
    KS/PSI (não guarda janela, é O(1) por amostra), então costuma ser o mais
    rápido para disparar — mas também o mais sensível a ruído se `delta`/
    `threshold` não forem calibrados.
    """

    name = "page_hinkley"

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999) -> None:
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._mean = 0.0
        self._n = 0
        self._sum = 0.0
        self._min_sum = 0.0

    def _update(self, x: float) -> bool:
        self._n += 1
        self._mean = self._mean + (x - self._mean) / self._n
        self._sum = self.alpha * self._sum + (x - self._mean - self.delta)
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        return ph > self.threshold

    def reset(self) -> None:
        super().reset()
        self._mean = 0.0
        self._n = 0
        self._sum = 0.0
        self._min_sum = 0.0


# ════════════════════════════════════════════════════════════════
# CUSUM
# ════════════════════════════════════════════════════════════════

class CUSUMDriftDetector(BaseDriftDetector):
    """Soma cumulativa clássica sobre o desvio em relação à média de
    referência, com "slack" (drift) para não acumular ruído.
    """

    name = "cusum"

    def __init__(self, reference: np.ndarray, drift_slack: float | None = None, threshold: float = 5.0) -> None:
        super().__init__()
        reference = np.asarray(reference, dtype=float)
        self.ref_mean = float(reference.mean())
        self.ref_std = float(reference.std()) or 1e-6
        self.drift_slack = drift_slack if drift_slack is not None else 0.5 * self.ref_std
        self.threshold = threshold * self.ref_std
        self._pos = 0.0
        self._neg = 0.0

    def _update(self, x: float) -> bool:
        dev = x - self.ref_mean
        self._pos = max(0.0, self._pos + dev - self.drift_slack)
        self._neg = min(0.0, self._neg + dev + self.drift_slack)
        return self._pos > self.threshold or -self._neg > self.threshold

    def reset(self) -> None:
        super().reset()
        self._pos = 0.0
        self._neg = 0.0


# ════════════════════════════════════════════════════════════════
# ADWIN-lite (janela adaptativa simplificada, sem dependência externa)
# ════════════════════════════════════════════════════════════════

class ADWINLiteDetector(BaseDriftDetector):
    """Versão simplificada do espírito do ADWIN: mantém uma janela deslizante
    e testa, a cada passo, se dividir a janela em duas metades revela uma
    diferença de médias grande o suficiente (normalizada pelo desvio padrão
    combinado). Não é o algoritmo ADWIN completo (que usa buckets exponenciais
    e um critério estatístico formal por corte), mas captura a mesma ideia de
    "janela que encolhe quando muda" com poucas linhas e zero dependências.

    Se precisar do ADWIN "de verdade", instale `river` (`pip install river
    --break-system-packages`) e use `river.drift.ADWIN`.
    """

    name = "adwin_lite"

    def __init__(self, max_window: int = 200, min_subwindow: int = 15, threshold: float = 3.0) -> None:
        super().__init__()
        self.max_window = max_window
        self.min_subwindow = min_subwindow
        self.threshold = threshold
        self._window: list[float] = []

    def _update(self, x: float) -> bool:
        self._window.append(x)
        if len(self._window) > self.max_window:
            self._window.pop(0)

        n = len(self._window)
        if n < 2 * self.min_subwindow:
            return False

        arr = np.array(self._window)
        drift_found = False
        # testa alguns pontos de corte (não todos, por custo) espaçados no meio da janela
        cut_points = range(self.min_subwindow, n - self.min_subwindow, max(1, n // 10))
        for cut in cut_points:
            w0, w1 = arr[:cut], arr[cut:]
            n0, n1 = len(w0), len(w1)
            pooled_std = np.sqrt((w0.var() * n0 + w1.var() * n1) / (n0 + n1)) or 1e-6
            eps = pooled_std * np.sqrt((1 / n0 + 1 / n1))
            if abs(w0.mean() - w1.mean()) > self.threshold * eps:
                drift_found = True
                # descarta a parte antiga (antes do corte) — janela "adapta"
                self._window = list(arr[cut:])
                break

        return drift_found

    def reset(self) -> None:
        super().reset()
        self._window.clear()


# ════════════════════════════════════════════════════════════════
# Fábrica padrão de detectores para o benchmark
# ════════════════════════════════════════════════════════════════

def default_detectors(reference: np.ndarray) -> dict[str, BaseDriftDetector]:
    """Conjunto padrão de detectores, todos calibrados a partir da mesma
    janela de referência (ex.: erros de reconstrução do período de treino,
    que representa "operação normal").
    """
    return {
        "ks_test": KSDriftDetector(reference, window_size=50, alpha=0.01),
        "ks_test_w100": KSDriftDetector(reference, window_size=100, alpha=0.01),
        "psi": PSIDriftDetector(reference, window_size=50, threshold=0.2),
        "page_hinkley": PageHinkleyDetector(delta=0.005, threshold=50.0),
        "cusum": CUSUMDriftDetector(reference, threshold=5.0),
        "adwin_lite": ADWINLiteDetector(max_window=200, min_subwindow=15, threshold=3.0),
    }