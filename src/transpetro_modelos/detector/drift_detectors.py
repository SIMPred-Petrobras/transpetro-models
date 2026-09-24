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
    - CalibratedKSDetector   : KS com limiar do D calibrado na referência + persistência
                               (o detector adotado no monitor de produção; lê/grava drift_ref.json)

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
# KS calibrado (limiar empírico do estatístico D + persistência)
# ════════════════════════════════════════════════════════════════

class CalibratedKSDetector(BaseDriftDetector):
    """KS por feature com limiar do estatístico D calibrado na própria referência.

    Diferenças para o KSDriftDetector (p-valor):
      - não usa p-valor: em série autocorrelacionada ele sai minúsculo para
        diferenças triviais. O limiar de cada feature é o MAIOR D observado ao
        comparar cada janela da própria referência com a referência inteira
        ("diferente demais" = mais diferente do que o normal já é de si mesmo);
      - janelas não sobrepostas de `window_size` pontos (288 = 1 dia a 5 min);
      - persistência: só dispara quando `k_consecutive` das últimas
        `persistence_window` janelas têm pelo menos uma feature acima do limiar
        (padrão 3 de 5 dias; `persistence_window=None` = k janelas seguidas).
        Filtra transientes (mudança de conceito precisa ser perene) sem
        depender do alinhamento das janelas: no drift real do B-8802B os dias
        acima do limiar vêm intercalados, e a regra de dias SEGUIDOS variava de
        3,4 a 75,7 dias conforme o alinhamento; 3-de-5 fica entre 3,4 e 8,4
        dias, com zero falso disparo no controle sem drift;
      - informa quais features causaram o disparo (`last_drift_features`).

    O estado calibrado (amostras de referência + limiares) é o mesmo formato
    do `drift_ref.json` gravado nos bundles por `scripts/monitor_drift.py
    --make-drift-ref`: use `from_drift_ref()` / `to_drift_ref()`.
    """

    name = "ks_calibrado"

    def __init__(
        self,
        reference: np.ndarray | pd.DataFrame | None = None,
        window_size: int = 288,
        k_consecutive: int = 3,
        n_reference: int = 5000,
        persistence_window: int | None = 5,
        feature_names: list[str] | None = None,
        seed: int = 0,
        *,
        samples: list[np.ndarray] | None = None,
        d_crit: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.k_consecutive = k_consecutive
        # None = k janelas CONSECUTIVAS; n = k janelas acima dentre as últimas n (tolera dia intercalado)
        self.persistence_window = persistence_window

        if samples is not None and d_crit is not None:
            self.samples = [np.asarray(v, dtype=float) for v in samples]
            self.d_crit = [float(v) for v in d_crit]
            self.feature_names = list(feature_names) if feature_names else [f"x{j}" for j in range(len(self.samples))]
        else:
            if reference is None:
                raise ValueError("informe `reference` ou (`samples` e `d_crit`).")
            if isinstance(reference, pd.DataFrame) and feature_names is None:
                feature_names = [str(c) for c in reference.columns]
            ref = np.asarray(reference, dtype=float)
            if ref.ndim == 1:
                ref = ref.reshape(-1, 1)
            self.feature_names = list(feature_names) if feature_names else [f"x{j}" for j in range(ref.shape[1])]
            rng = np.random.default_rng(seed)
            self.samples, self.d_crit = [], []
            for j in range(ref.shape[1]):
                col = ref[:, j][np.isfinite(ref[:, j])]
                if len(col) < 2 * window_size:
                    raise ValueError(
                        f"referência de '{self.feature_names[j]}' tem {len(col)} pontos; "
                        f"são necessários pelo menos {2 * window_size} (2 janelas)."
                    )
                sample = rng.choice(col, size=min(len(col), n_reference), replace=False)
                dmax = max(
                    stats.ks_2samp(sample, col[i:i + window_size]).statistic
                    for i in range(0, len(col) - window_size, window_size)
                )
                self.samples.append(sample)
                self.d_crit.append(float(dmax))

        self._buffer: list[np.ndarray] = []
        self._runs: list[set[str]] = []
        self.last_d: np.ndarray | None = None
        self.last_drift_features: list[str] = []

    @classmethod
    def from_drift_ref(cls, drift_ref: dict | str) -> "CalibratedKSDetector":
        """Constrói o detector a partir de um drift_ref.json (dict ou caminho)."""
        if not isinstance(drift_ref, dict):
            import json
            from pathlib import Path
            drift_ref = json.loads(Path(drift_ref).read_text())
        names = list(drift_ref["sensors"])
        return cls(
            window_size=int(drift_ref["win"]),
            k_consecutive=int(drift_ref["k_consec"]),
            persistence_window=drift_ref.get("n_window"),
            feature_names=names,
            samples=[drift_ref["sensors"][c]["sample"] for c in names],
            d_crit=[drift_ref["sensors"][c]["d_crit"] for c in names],
        )

    def to_drift_ref(self, reference_window: list[str] | None = None, n_keep: int = 2000) -> dict:
        """Serializa no formato do drift_ref.json dos bundles."""
        rng = np.random.default_rng(0)
        sensors = {}
        for name, sample, dc in zip(self.feature_names, self.samples, self.d_crit):
            keep = rng.choice(sample, size=min(len(sample), n_keep), replace=False)
            sensors[name] = {"d_crit": float(dc), "sample": [round(float(v), 4) for v in keep]}
        out = {"reference_window": reference_window, "win": self.window_size,
               "k_consec": self.k_consecutive, "sensors": sensors}
        if self.persistence_window is not None:
            out["n_window"] = self.persistence_window
        return out

    def _update(self, x: float) -> bool:
        return self.update(x)

    def update(self, x) -> bool:
        """Recebe um ponto (escalar ou vetor com uma posição por feature)."""
        if self._in_alarm:
            return False
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != len(self.samples):
            raise ValueError(f"esperado {len(self.samples)} feature(s), recebido {arr.size}.")
        self._buffer.append(arr)
        if len(self._buffer) < self.window_size:
            return False

        current = np.asarray(self._buffer, dtype=float)
        self._buffer.clear()
        d = np.zeros(len(self.samples))
        above: set[str] = set()
        for j, (sample, dc) in enumerate(zip(self.samples, self.d_crit)):
            cur = current[:, j][np.isfinite(current[:, j])]
            if len(cur) == 0:
                continue
            d[j] = stats.ks_2samp(sample, cur).statistic
            if d[j] > dc:
                above.add(self.feature_names[j])
        self.last_d = d

        if self.persistence_window is None:
            self._runs = self._runs + [above] if above else []
            if len(self._runs) < self.k_consecutive:
                return False
            recent = self._runs[-self.k_consecutive:]
        else:
            self._runs = (self._runs + [above])[-self.persistence_window:]
            recent = [r for r in self._runs if r]
            if len(recent) < self.k_consecutive:
                return False
        common = set.intersection(*recent) or set().union(*recent)
        self.last_drift_features = sorted(common)
        self._runs = []
        self._in_alarm = True
        return True

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._runs = []
        self.last_d = None


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
        "ks_calibrado": CalibratedKSDetector(ref),
        "psi": PSIDriftDetector(ref, window_size=50, threshold=0.2),
        "page_hinkley": PageHinkleyDetector(delta=0.005, threshold=50.0),
        "cusum": CUSUMDriftDetector(ref, threshold=5.0),
        "adwin_lite": ADWINLiteDetector(max_window=200, min_subwindow=15, threshold=3.0),
    }