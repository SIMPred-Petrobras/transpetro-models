"""
drift_detectors.py  (versão corrigida)
==================================
Detectores de concept drift para série temporal de scores/erros de reconstrução
(ou qualquer feature contínua). Interface online comum: `update(x) -> bool`.

Detectores:
    - KSDriftDetector        : KS (p-valor) janela atual vs referência
    - PSIDriftDetector       : Population Stability Index
    - PageHinkleyDetector    : mudança de média, padronizado, média auto-adaptativa
    - CUSUMDriftDetector     : soma cumulativa padronizada, média da referência
    - ADWINLiteDetector      : janela adaptativa simplificada
    - CalibratedKSDetector   : KS com limiar D calibrado na referência + persistência
                               (lê/grava drift_ref.json)

O QUE MUDOU EM RELAÇÃO À VERSÃO ANTERIOR
----------------------------------------
 1. PSI: bordas externas = ±inf. Antes, valores acima do máximo da referência
    (justamente o drift) eram descartados pelo np.histogram.
 2. PSI: janela padrão = 1 dia (era 50 pts, onde o PSI "normal" já vale ~0,18),
    5 buckets, limiar calibrado na referência (piso 0,2) e confirmação.
 3. KS / PSI / ADWIN: `stride` (antes testavam a cada ponto, sem correção) e
    confirmação por testes consecutivos. alpha padrão do KS = 1e-4.
 4. Page-Hinkley e CUSUM: padronizados (mediana/MAD da referência), parâmetros em
    unidades de desvio, limiar `h` calibrado por bootstrap em blocos para uma
    taxa de falso alarme alvo (padrão: ~5% de chance de falso disparo em 90 dias).
    O PH agora é auto-referenciado (média móvel exponencial lenta), então deixa de
    ser o mesmo detector do CUSUM.
 5. Todos aceitam `direction`: "up" (só sobe, o padrão no build_detector),
    "down" ou "both". Queda de erro não deve disparar retreino.
 6. CalibratedKS: d_crit por leave-block-out (sem vazamento), margem de segurança
    (padrão 1,15) e `direction`. drift_ref.json antigo continua carregando
    (assume "both").
 7. `last_score` (estatística contínua) em todos, para plotar e varrer limiar.
 8. `rearm()`: reabilita o detector SEM zerar o estado acumulado. Use quando o
    portão do walk-forward não abriu (o drift foi "adiado", não "descartado").
 9. build_detector aplica log no erro por padrão (`transform="log"`): reduz a cauda
    pesada. Passe transform="none" para o comportamento antigo.
10. Teste sintético embutido: `python drift_detectors.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats

_DIRECTIONS = ("up", "down", "both")


def _check_direction(direction: str) -> str:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction deve ser um de {_DIRECTIONS}, recebido '{direction}'.")
    return direction


def _ks_stat(ref: np.ndarray, cur: np.ndarray, direction: str) -> tuple[float, float]:
    """(D, p-valor). 'up' = a janela atual tende a valores MAIORES que a referência."""
    if direction == "both":
        r = stats.ks_2samp(ref, cur)
    elif direction == "up":
        r = stats.ks_2samp(ref, cur, alternative="greater")
    else:
        r = stats.ks_2samp(ref, cur, alternative="less")
    return float(r.statistic), float(r.pvalue)


def _robust_center_scale(ref: np.ndarray) -> tuple[float, float]:
    mu = float(np.median(ref))
    sd = 1.4826 * float(np.median(np.abs(ref - mu)))
    if sd <= 0:
        sd = float(ref.std())
    if sd <= 0:
        sd = 1e-6
    return mu, sd


# ════════════════════════════════════════════════════════════════
# Interface comum
# ════════════════════════════════════════════════════════════════

class BaseDriftDetector(ABC):
    """Consome um valor por vez; retorna True quando dispara.

    Depois de disparar fica "em alarme" (ignora entradas) até `reset()` ou `rearm()`:
      - reset(): zera tudo (referência de curto prazo, acumuladores);
      - rearm(): só reabilita; mantém o estado acumulado (drift adiado pelo portão).
    """

    name: str = "base"
    multivariate: bool = False

    def __init__(self) -> None:
        self._in_alarm = False
        self._last_score = float("nan")

    @property
    def last_score(self) -> float:
        """Última estatística calculada (escala depende do detector)."""
        return self._last_score

    @abstractmethod
    def _update(self, x) -> bool:
        """Lógica específica. Retorna True se disparou agora."""

    def update(self, x) -> bool:
        if self._in_alarm:
            return False
        x = np.asarray(x, dtype=float).reshape(-1) if self.multivariate else float(x)
        fired = bool(self._update(x))
        if fired:
            self._in_alarm = True
        return fired

    def rearm(self) -> None:
        self._in_alarm = False

    def reset(self) -> None:
        self._in_alarm = False
        self._last_score = float("nan")


# ════════════════════════════════════════════════════════════════
# Kolmogorov-Smirnov (p-valor)
# ════════════════════════════════════════════════════════════════

class KSDriftDetector(BaseDriftDetector):
    """KS de duas amostras, por feature, com decisão multivariada.

    Dispara quando >= `min_features` features têm p < alpha em `confirm_tests`
    testes consecutivos (testes espaçados por `stride`). Referência 1D ou 2D.
    """

    name = "ks_test"
    multivariate = True

    def __init__(
        self,
        reference: np.ndarray | pd.DataFrame,
        window_size: int = 50,
        alpha: float = 1e-4,
        stride: int | None = None,
        min_features: int | None = None,
        min_drift_fraction: float = 0.10,
        direction: str = "both",
        confirm_tests: int = 3,
    ) -> None:
        super().__init__()
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if ref.ndim != 2:
            raise ValueError("reference deve ser 1D ou 2D.")

        self.reference = ref
        self._ref_cols = [c[np.isfinite(c)] for c in ref.T]
        self.window_size = window_size
        self.alpha = alpha
        self.stride = stride if stride else max(1, window_size // 5)
        self.direction = _check_direction(direction)
        self.confirm_tests = max(1, confirm_tests)

        nf = ref.shape[1]
        if min_features is None:
            min_features = max(1, int(np.ceil(nf * min_drift_fraction)))
        self.min_features = min(min_features, nf)

        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._since = 0
        self._hits = 0
        self.last_p_values: np.ndarray | None = None
        self.last_drift_mask: np.ndarray | None = None

    def _update(self, x: np.ndarray) -> bool:
        if x.size != self.reference.shape[1]:
            raise ValueError(f"KS recebeu {x.size} feature(s), referência tem {self.reference.shape[1]}.")
        self._buffer.append(x.copy())
        self._since += 1
        if len(self._buffer) < self.window_size or self._since < self.stride:
            return False
        self._since = 0

        cur = np.asarray(self._buffer, dtype=float)
        p = np.ones(self.reference.shape[1])
        for j, ref_j in enumerate(self._ref_cols):
            cur_j = cur[:, j][np.isfinite(cur[:, j])]
            if len(ref_j) and len(cur_j):
                p[j] = _ks_stat(ref_j, cur_j, self.direction)[1]

        mask = p < self.alpha
        self.last_p_values, self.last_drift_mask = p, mask
        self._last_score = float(-np.log10(max(float(p.min()), 1e-300)))
        self._hits = self._hits + 1 if int(mask.sum()) >= self.min_features else 0
        return self._hits >= self.confirm_tests

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._since = 0
        self._hits = 0
        self.last_p_values = None
        self.last_drift_mask = None


# ════════════════════════════════════════════════════════════════
# PSI
# ════════════════════════════════════════════════════════════════

class PSIDriftDetector(BaseDriftDetector):
    """PSI com bordas em quantis da referência e extremos abertos (±inf).

    O limiar é calibrado: max(threshold, margin x maior PSI entre janelas da
    própria referência). O 0,2 clássico só vale para amostras grandes; com janela
    pequena o PSI "normal" já fica perto dele.
    """

    name = "psi"

    def __init__(
        self,
        reference: np.ndarray,
        window_size: int = 288,
        n_buckets: int = 5,
        threshold: float = 0.2,
        stride: int | None = None,
        eps: float = 1e-4,
        direction: str = "up",
        calibrate: bool = True,
        margin: float = 1.25,
        confirm_tests: int = 2,
    ) -> None:
        super().__init__()
        ref = np.asarray(reference, dtype=float)
        ref = ref[np.isfinite(ref)]
        if len(ref) < 20:
            raise ValueError("PSI: referência com menos de 20 pontos.")

        inner = np.unique(np.quantile(ref, np.linspace(0, 1, n_buckets + 1)[1:-1]))
        if len(inner) < 2:  # quase constante
            lo, hi = ref.min() - eps, ref.max() + eps
            inner = np.linspace(lo, hi, n_buckets + 1)[1:-1]
        self._inner = inner
        self.eps = eps
        self.ref_frac = self._frac(ref)
        self.ref_median = float(np.median(ref))
        self.window_size = window_size
        self.stride = stride if stride else max(1, window_size // 5)
        self.direction = _check_direction(direction)
        self.confirm_tests = max(1, confirm_tests)

        self.threshold = threshold
        if calibrate and len(ref) >= 2 * window_size:
            step = max(1, window_size // 2)
            psis = [self._psi(ref[i:i + window_size]) for i in range(0, len(ref) - window_size + 1, step)]
            self.threshold = max(threshold, margin * max(psis))

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._since = 0
        self._hits = 0

    def _frac(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._inner, values, side="right")
        counts = np.bincount(idx, minlength=len(self._inner) + 1)
        return counts / max(counts.sum(), 1)

    def _psi(self, values: np.ndarray) -> float:
        ref = np.clip(self.ref_frac, self.eps, None)
        cur = np.clip(self._frac(values), self.eps, None)
        return float(np.sum((cur - ref) * np.log(cur / ref)))

    def _update(self, x: float) -> bool:
        self._buffer.append(x)
        self._since += 1
        if len(self._buffer) < self.window_size or self._since < self.stride:
            return False
        self._since = 0

        vals = np.asarray(self._buffer, dtype=float)
        psi = self._psi(vals)
        med = float(np.median(vals))
        sentido_ok = (
            self.direction == "both"
            or (self.direction == "up" and med > self.ref_median)
            or (self.direction == "down" and med < self.ref_median)
        )
        self._last_score = psi
        self._hits = self._hits + 1 if (psi > self.threshold and sentido_ok) else 0
        return self._hits >= self.confirm_tests

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._since = 0
        self._hits = 0


# ════════════════════════════════════════════════════════════════
# CUSUM e Page-Hinkley (padronizados, limiar calibrado)
# ════════════════════════════════════════════════════════════════

def _seq_peak(z: np.ndarray, k: float, direction: str, m0: float, rate: float) -> float:
    """Maior valor da estatística sequencial ao longo de z (usado na calibração)."""
    pos = neg = peak = 0.0
    m = m0
    up, down = direction in ("up", "both"), direction in ("down", "both")
    for v in z:
        if up:
            pos = max(0.0, pos + v - m - k)
        if down:
            neg = max(0.0, neg + m - v - k)
        s = max(pos, neg)
        if s > peak:
            peak = s
        if rate > 0:
            m += rate * (v - m)
    return peak


class _SeqDetector(BaseDriftDetector):
    """Base de CUSUM/Page-Hinkley: soma cumulativa sobre z = (x - mediana)/MAD_sigma.

    s+ = max(0, s+ + z - m - k);  s- = max(0, s- + m - z - k);  dispara se s > h.
      - CUSUM: m = 0 fixo (média da referência).
      - PH:    m = média móvel exponencial lenta de z (auto-referenciado).
    `h` None => calibrado por bootstrap em blocos: h = quantil `target_quantile` do
    pico da estatística em séries ressintetizadas da referência com `horizon_days`
    dias (preserva a autocorrelação dentro do bloco).
    """

    def __init__(
        self,
        reference: np.ndarray,
        k: float,
        h: float | None,
        direction: str,
        mean_rate: float,
        samples_per_day: int = 288,
        horizon_days: int = 90,
        n_boot: int = 20,
        target_quantile: float = 0.95,
        margin: float = 1.0,
        h_min: float = 3.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        ref = np.asarray(reference, dtype=float)
        ref = ref[np.isfinite(ref)]
        if len(ref) < 20:
            raise ValueError(f"{self.name}: referência com menos de 20 pontos.")
        self.direction = _check_direction(direction)
        self.mu, self.sd = _robust_center_scale(ref)
        z_ref = (ref - self.mu) / self.sd
        self.k = k
        self.mean_rate = mean_rate
        self._m0 = float(z_ref.mean()) if mean_rate > 0 else 0.0

        if h is None:
            rng = np.random.default_rng(seed)
            b = max(5, min(samples_per_day, len(z_ref) // 4))
            horizon = horizon_days * samples_per_day
            n_blocks = int(np.ceil(horizon / b))
            peaks = []
            for _ in range(n_boot):
                starts = rng.integers(0, len(z_ref) - b + 1, size=n_blocks)
                z = np.concatenate([z_ref[s:s + b] for s in starts])[:horizon]
                peaks.append(_seq_peak(z, k, self.direction, self._m0, mean_rate))
            h = max(h_min, margin * float(np.quantile(peaks, target_quantile)))
        self.h = float(h)

        self._pos = self._neg = 0.0
        self._m = self._m0

    def _update(self, x: float) -> bool:
        z = (x - self.mu) / self.sd
        m = self._m
        self._pos = max(0.0, self._pos + z - m - self.k)
        self._neg = max(0.0, self._neg + m - z - self.k)
        if self.mean_rate > 0:
            self._m = m + self.mean_rate * (z - m)
        if self.direction == "up":
            s = self._pos
        elif self.direction == "down":
            s = self._neg
        else:
            s = max(self._pos, self._neg)
        self._last_score = s
        return s > self.h

    def reset(self) -> None:
        super().reset()
        self._pos = self._neg = 0.0
        self._m = self._m0


class CUSUMDriftDetector(_SeqDetector):
    """CUSUM padronizado (k e h em desvios-padrão robustos da referência)."""

    name = "cusum"

    def __init__(self, reference: np.ndarray, k: float = 0.5, h: float | None = None,
                 direction: str = "up", **kw) -> None:
        super().__init__(reference, k=k, h=h, direction=direction, mean_rate=0.0, **kw)


class PageHinkleyDetector(_SeqDetector):
    """Page-Hinkley padronizado, com média móvel exponencial (constante de tempo
    ~`tau_days` dias). Pega mudanças relativas ao passado recente; mudanças lentas
    demais são absorvidas pela média, que é o comportamento esperado do PH."""

    name = "page_hinkley"

    def __init__(self, reference: np.ndarray, delta: float = 0.25, h: float | None = None,
                 direction: str = "up", tau_days: float = 7.0, samples_per_day: int = 288, **kw) -> None:
        super().__init__(reference, k=delta, h=h, direction=direction,
                         mean_rate=1.0 / (tau_days * samples_per_day),
                         samples_per_day=samples_per_day, **kw)


# ════════════════════════════════════════════════════════════════
# ADWIN-lite
# ════════════════════════════════════════════════════════════════

class ADWINLiteDetector(BaseDriftDetector):
    """Janela adaptativa simplificada (não é o ADWIN completo).

    Testa alguns pontos de corte a cada `stride` amostras. Dispara se a diferença
    de médias (recente - antiga) passa de `threshold` erros-padrão E de
    `min_effect` desvios da referência (significância prática: com muitos pontos
    qualquer diferença minúscula passa no teste estatístico). O erro-padrão é
    inflado pela autocorrelação lag-1 da referência (tamanho efetivo de amostra).
    """

    name = "adwin_lite"

    def __init__(
        self,
        reference: np.ndarray | None = None,
        max_window: int = 864,
        min_subwindow: int = 72,
        threshold: float = 4.0,
        stride: int | None = None,
        min_effect: float = 0.5,
        direction: str = "up",
        n_cuts: int = 10,
    ) -> None:
        super().__init__()
        self.max_window = max_window
        self.min_subwindow = min_subwindow
        self.threshold = threshold
        self.stride = stride if stride else max(1, min_subwindow // 4)
        self.min_effect = min_effect
        self.direction = _check_direction(direction)
        self.n_cuts = n_cuts
        self.ref_sd: float | None = None
        self.inflate = 1.0  # sqrt((1+rho)/(1-rho)): corrige o erro-padrão por autocorrelação
        if reference is not None:
            ref = np.asarray(reference, dtype=float)
            ref = ref[np.isfinite(ref)]
            if len(ref) >= 3:
                self.ref_sd = _robust_center_scale(ref)[1]
                rho = float(np.corrcoef(ref[:-1], ref[1:])[0, 1])
                if np.isfinite(rho):
                    rho = min(max(rho, 0.0), 0.99)
                    self.inflate = float(np.sqrt((1 + rho) / (1 - rho)))
        self._window: deque[float] = deque(maxlen=max_window)
        self._since = 0

    def _update(self, x: float) -> bool:
        self._window.append(x)
        self._since += 1
        n = len(self._window)
        if n < 2 * self.min_subwindow or self._since < self.stride:
            return False
        self._since = 0

        arr = np.asarray(self._window, dtype=float)
        cuts = np.unique(np.linspace(self.min_subwindow, n - self.min_subwindow, self.n_cuts).astype(int))
        best_z, hit_cut = 0.0, None
        for cut in cuts:
            w0, w1 = arr[:cut], arr[cut:]
            n0, n1 = len(w0), len(w1)
            scale = self.ref_sd or np.sqrt((w0.var() * n0 + w1.var() * n1) / (n0 + n1)) or 1e-6
            diff = float(w1.mean() - w0.mean())
            signed = diff if self.direction == "up" else -diff if self.direction == "down" else abs(diff)
            z = signed / (scale * self.inflate * np.sqrt(1 / n0 + 1 / n1))
            best_z = max(best_z, z)
            if z > self.threshold and abs(diff) > self.min_effect * scale and hit_cut is None:
                hit_cut = int(cut)
        self._last_score = best_z
        if hit_cut is not None:
            self._window = deque(arr[hit_cut:], maxlen=self.max_window)  # descarta a parte antiga
            return True
        return False

    def reset(self) -> None:
        super().reset()
        self._window.clear()
        self._since = 0


# ════════════════════════════════════════════════════════════════
# KS calibrado (limiar empírico de D + persistência)
# ════════════════════════════════════════════════════════════════

class CalibratedKSDetector(BaseDriftDetector):
    """KS por feature com limiar do estatístico D calibrado na própria referência.

      - sem p-valor (em série autocorrelacionada ele sai minúsculo);
      - d_crit = margin x maior D de um bloco da referência contra o RESTO da
        referência (leave-block-out; antes o bloco estava dentro da amostra de
        comparação e o D saía viesado para baixo);
      - janelas não sobrepostas de `window_size` pontos (288 = 1 dia a 5 min);
      - persistência: `k_consecutive` das últimas `persistence_window` janelas com
        alguma feature acima do limiar (`persistence_window=None` = k seguidas);
      - `direction="up"` testa só aumento (use para erro de reconstrução);
      - informa as features que causaram o disparo (`last_drift_features`).

    Formato compatível com drift_ref.json (`from_drift_ref` / `to_drift_ref`).
    """

    name = "ks_calibrado"
    multivariate = True

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
        direction: str = "both",
        margin: float = 1.15,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.k_consecutive = k_consecutive
        self.persistence_window = persistence_window
        self.direction = _check_direction(direction)

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
                dmax = 0.0
                for i in range(0, len(col) - window_size + 1, window_size):
                    blk = col[i:i + window_size]
                    rest = np.concatenate([col[:i], col[i + window_size:]])
                    dmax = max(dmax, _ks_stat(rest, blk, self.direction)[0])
                self.samples.append(sample)
                self.d_crit.append(float(dmax * margin))

        self._buffer: list[np.ndarray] = []
        self._runs: list[set[str]] = []
        self.last_d: np.ndarray | None = None
        self.last_drift_features: list[str] = []

    @classmethod
    def from_drift_ref(cls, drift_ref: dict | str, direction: str | None = None) -> "CalibratedKSDetector":
        """Constrói a partir de um drift_ref.json (dict ou caminho).
        Sem a chave 'direction' no arquivo assume 'both' (limiares antigos eram bilaterais)."""
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
            direction=direction or drift_ref.get("direction", "both"),
        )

    def to_drift_ref(self, reference_window: list[str] | None = None, n_keep: int = 2000) -> dict:
        rng = np.random.default_rng(0)
        sensors = {}
        for name, sample, dc in zip(self.feature_names, self.samples, self.d_crit):
            keep = rng.choice(sample, size=min(len(sample), n_keep), replace=False)
            sensors[name] = {"d_crit": float(dc), "sample": [round(float(v), 4) for v in keep]}
        out = {"reference_window": reference_window, "win": self.window_size,
               "k_consec": self.k_consecutive, "direction": self.direction, "sensors": sensors}
        if self.persistence_window is not None:
            out["n_window"] = self.persistence_window
        return out

    def _update(self, arr: np.ndarray) -> bool:
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
            d[j] = _ks_stat(sample, cur, self.direction)[0]
            if d[j] > dc:
                above.add(self.feature_names[j])
        self.last_d = d
        self._last_score = float(max(dj / max(dc, 1e-12) for dj, dc in zip(d, self.d_crit)))

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
        return True

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._runs = []
        self.last_d = None

    def scan(self, frame: pd.DataFrame | np.ndarray, index=None) -> tuple[pd.DataFrame, list]:
        """Passa uma série inteira pelo detector, como o monitor faz em produção.
        Retorna (d_diario, disparos); após cada disparo o detector é resetado."""
        if isinstance(frame, pd.DataFrame):
            index = frame.index if index is None else index
            X = frame[self.feature_names].to_numpy(dtype=float)
        else:
            X = np.asarray(frame, dtype=float)
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            index = pd.RangeIndex(len(X)) if index is None else index
        self.reset()
        linhas, instantes, disparos = [], [], []
        for t, x in zip(index, X):
            fired = self.update(x)
            if not self._buffer:
                linhas.append(self.last_d.copy())
                instantes.append(t)
            if fired:
                disparos.append((t, list(self.last_drift_features)))
                self.reset()
        d_diario = pd.DataFrame(linhas, index=pd.Index(instantes, name=getattr(index, "name", None)),
                                columns=self.feature_names)
        return d_diario, disparos


# ════════════════════════════════════════════════════════════════
# Fábrica
# ════════════════════════════════════════════════════════════════

DETECTOR_NAMES = [
    "ks_test", "ks_test_w100", "ks_calibrado",
    "psi", "page_hinkley", "cusum", "adwin_lite",
]


class _LogInput(BaseDriftDetector):
    """Aplica log(max(x,0)+eps) antes de repassar ao detector interno."""

    def __init__(self, inner: BaseDriftDetector, eps: float) -> None:
        super().__init__()
        self.inner = inner
        self.eps = eps
        self.name = inner.name

    @property
    def last_score(self) -> float:
        return self.inner.last_score

    def _update(self, x: float) -> bool:
        return self.inner.update(float(np.log(max(x, 0.0) + self.eps)))

    def rearm(self) -> None:
        super().rearm()
        self.inner.rearm()

    def reset(self) -> None:
        super().reset()
        self.inner.reset()

    def __getattr__(self, item):  # window_size, k_consecutive, min_subwindow, threshold, ...
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)


def build_detector(
    name: str,
    reference: np.ndarray,
    samples_per_day: int | None = None,
    *,
    direction: str = "up",
    transform: str = "log",
) -> BaseDriftDetector:
    """Instancia UM detector (sem construir os outros).

    samples_per_day : amostras por dia (janela de 1 dia do KS calibrado/PSI; escala do
                      ADWIN e da calibração de CUSUM/PH). Default 288.
    direction       : "up" (padrão: só aumento de erro), "down" ou "both".
    transform       : "log" (padrão, reduz cauda pesada) ou "none".
    """
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 1:
        raise ValueError("build_detector(): a referência deve ser 1D (erro de reconstrução).")
    ref = ref[np.isfinite(ref)]
    spd = int(samples_per_day) if samples_per_day else 288

    eps = None
    if transform == "log":
        eps = 1e-3 * (float(np.median(np.abs(ref))) or 1.0)
        ref = np.log(np.clip(ref, 0.0, None) + eps)
    elif transform != "none":
        raise ValueError("transform deve ser 'log' ou 'none'.")

    d = direction
    factories = {
        "ks_test":      lambda: KSDriftDetector(ref, window_size=50, direction=d),
        "ks_test_w100": lambda: KSDriftDetector(ref, window_size=100, direction=d),
        "ks_calibrado": lambda: CalibratedKSDetector(ref, window_size=spd, direction=d),
        "psi":          lambda: PSIDriftDetector(ref, window_size=spd, direction=d),
        "page_hinkley": lambda: PageHinkleyDetector(ref, direction=d, samples_per_day=spd),
        "cusum":        lambda: CUSUMDriftDetector(ref, direction=d, samples_per_day=spd),
        "adwin_lite":   lambda: ADWINLiteDetector(ref, max_window=3 * spd,
                                                   min_subwindow=max(15, spd // 4), direction=d),
    }
    if name not in factories:
        raise KeyError(f"detector desconhecido: '{name}'. Opções: {list(factories)}")
    det = factories[name]()
    return _LogInput(det, eps) if eps is not None else det


def default_detectors(reference: np.ndarray | pd.DataFrame, **kw) -> dict[str, BaseDriftDetector]:
    """Todos os detectores de DETECTOR_NAMES (referência 1D). Os que não puderem ser
    construídos (ex.: KS calibrado com referência curta) são ignorados com aviso."""
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 1:
        raise ValueError(
            "default_detectors(): referência deve ser 1D. Para KS multifeature, "
            "instancie KSDriftDetector ou CalibratedKSDetector diretamente."
        )
    out: dict[str, BaseDriftDetector] = {}
    for n in DETECTOR_NAMES:
        try:
            out[n] = build_detector(n, ref, **kw)
        except ValueError as exc:
            print(f"[aviso] detector '{n}' ignorado: {exc}")
    return out


# ════════════════════════════════════════════════════════════════
# Teste sintético: falso alarme sem drift, atraso em degrau e rampa
# ════════════════════════════════════════════════════════════════

def _simular(n: int, spd: int, rng, shift_start: int | None = None, shift: float = 0.0, ramp: bool = False):
    """Erro log-normal com ruído AR(1) (autocorrelacionado, cauda pesada)."""
    e = rng.normal(size=n)
    z = np.empty(n)
    z[0] = e[0]
    for i in range(1, n):
        z[i] = 0.9 * z[i - 1] + np.sqrt(1 - 0.81) * e[i]
    lvl = np.zeros(n)
    if shift_start is not None:
        if ramp:
            lvl[shift_start:] = np.linspace(0, shift * 2, n - shift_start)
        else:
            lvl[shift_start:] = shift
    return 0.05 * np.exp(0.3 * z + lvl)


def benchmark_sintetico(spd: int = 288, ref_dias: int = 7, dias: int = 90, shift: float = 0.5,
                        seeds: int = 2, detectores: list[str] | None = None) -> pd.DataFrame:
    """Mede, por detector: falsos disparos/100 dias (sem drift) e atraso em dias (degrau/rampa).
    Cada disparo é seguido de reset(), como no walk-forward."""
    import time
    detectores = detectores or DETECTOR_NAMES
    n = dias * spd
    ini = (dias // 3) * spd
    linhas = []
    for nome in detectores:
        fa, atr_deg, atr_ram, t0 = [], [], [], time.time()
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            ref = _simular(ref_dias * spd, spd, rng)
            for cenario in ("normal", "degrau", "rampa"):
                serie = _simular(n, spd, rng,
                                 None if cenario == "normal" else ini, shift, cenario == "rampa")
                try:
                    det = build_detector(nome, ref, spd)
                except ValueError as exc:
                    print(f"[aviso] {nome}: {exc}")
                    break
                disparos = []
                for i, v in enumerate(serie):
                    if det.update(v):
                        disparos.append(i)
                        det.reset()
                if cenario == "normal":
                    fa.append(len(disparos) * 100.0 / dias)
                else:
                    pos = [i for i in disparos if i >= ini]
                    atraso = (pos[0] - ini) / spd if pos else np.nan
                    (atr_deg if cenario == "degrau" else atr_ram).append(atraso)
        if fa:
            linhas.append({"detector": nome, "falsos_por_100d": np.mean(fa),
                           "atraso_degrau_d": np.nanmean(atr_deg) if not np.all(np.isnan(atr_deg)) else np.nan,
                           "atraso_rampa_d": np.nanmean(atr_ram) if not np.all(np.isnan(atr_ram)) else np.nan,
                           "seg": round(time.time() - t0, 1)})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    print(benchmark_sintetico().to_string(index=False))