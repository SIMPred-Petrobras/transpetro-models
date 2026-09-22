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
    """KS univariado por feature, com agregação para decisão multivariada.

    A referência pode ser:
      - 1D: comportamento original, para uma série contínua;
      - 2D: matriz/DataFrame (amostras x features). Nesse caso, um KS de
        duas amostras é executado separadamente em cada feature e o detector
        dispara quando pelo menos `min_features` features apresentam
        p-value < alpha.

    Isso mantém o KS estatisticamente univariado por variável, mas permite
    uma decisão global de drift sobre um conjunto de variáveis.
    """

    name = "ks_test"

    def __init__(
        self,
        reference: np.ndarray | pd.DataFrame,
        window_size: int = 50,
        alpha: float = 0.01,
        stride: int = 1,
        min_features: int | None = None,
        min_drift_fraction: float = 0.10,
    ) -> None:
        super().__init__()

        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)

        if ref.ndim != 2:
            raise ValueError("reference deve ser 1D ou 2D.")

        self.reference = ref
        self.window_size = window_size
        self.alpha = alpha
        self.stride = stride
        self.min_drift_fraction = min_drift_fraction

        n_features = ref.shape[1]
        if min_features is None:
            min_features = max(1, int(np.ceil(n_features * min_drift_fraction)))
        self.min_features = min(min_features, n_features)

        self._buffer: list[np.ndarray] = []
        self._since_last_test = 0

        # Útil para auditoria: guardar o resultado do último teste.
        self.last_p_values: np.ndarray | None = None
        self.last_drift_mask: np.ndarray | None = None

    def _update(self, x: float) -> bool:
        # Mantém compatibilidade com a interface antiga para KS 1D.
        return self._update_vector(np.asarray([x], dtype=float))

    def update(self, x) -> bool:
        """Recebe escalar (modo antigo) ou vetor de features."""
        if self._in_alarm:
            return False

        arr = np.asarray(x, dtype=float).reshape(-1)

        if arr.size != self.reference.shape[1]:
            raise ValueError(
                f"KS recebeu {arr.size} feature(s), mas a referência tem "
                f"{self.reference.shape[1]}."
            )

        fired = self._update_vector(arr)
        if fired:
            self._in_alarm = True
        return fired

    def _update_vector(self, x: np.ndarray) -> bool:
        self._buffer.append(x.copy())
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        self._since_last_test += 1
        if len(self._buffer) < self.window_size or self._since_last_test < self.stride:
            return False

        self._since_last_test = 0

        current = np.asarray(self._buffer, dtype=float)
        p_values = np.ones(self.reference.shape[1], dtype=float)

        for j in range(self.reference.shape[1]):
            ref_j = self.reference[:, j]
            cur_j = current[:, j]

            # Remove NaN/inf para o KS não quebrar.
            ref_j = ref_j[np.isfinite(ref_j)]
            cur_j = cur_j[np.isfinite(cur_j)]

            if len(ref_j) == 0 or len(cur_j) == 0:
                p_values[j] = 1.0
                continue

            _, p_values[j] = stats.ks_2samp(ref_j, cur_j)

        drift_mask = p_values < self.alpha
        n_drift = int(drift_mask.sum())

        self.last_p_values = p_values
        self.last_drift_mask = drift_mask

        return n_drift >= self.min_features

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._since_last_test = 0
        self.last_p_values = None
        self.last_drift_mask = None


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

def default_detectors(reference: np.ndarray | pd.DataFrame) -> dict[str, BaseDriftDetector]:
    """Conjunto padrão de detectores.

    Para referência 2D (amostras x features), KS opera feature a feature.
    Os demais detectores continuam univariados e, portanto, exigem uma
    referência 1D.
    """
    ref = np.asarray(reference, dtype=float)

    if ref.ndim != 1:
        raise ValueError(
            "default_detectors(): os detectores PSI/Page-Hinkley/CUSUM/ADWIN "
            "continuam univariados. Para KS multifeature, instancie KSDriftDetector "
            "diretamente com a matriz de referência."
        )

    return {
        "ks_test": KSDriftDetector(ref, window_size=50, alpha=0.01),
        "ks_test_w100": KSDriftDetector(ref, window_size=100, alpha=0.01),
        "psi": PSIDriftDetector(ref, window_size=50, threshold=0.2),
        "page_hinkley": PageHinkleyDetector(delta=0.005, threshold=50.0),
        "cusum": CUSUMDriftDetector(ref, threshold=5.0),
        "adwin_lite": ADWINLiteDetector(max_window=200, min_subwindow=15, threshold=3.0),
    }