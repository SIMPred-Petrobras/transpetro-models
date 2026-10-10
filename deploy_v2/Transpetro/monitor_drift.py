# ARQUIVO GERADO por scripts/package_monitor.py a partir de src/transpetro_modelos/drift/monitor.py
# e drift/detectors.py + drift/residuo.py. Não edite aqui: edite a origem e rode o gerador.
"""
Monitor de drift do SIMPred — calcula, semana a semana, os indicadores da política de retreino
(docs/politica_retreino.md) a partir do CSV que o próprio pacote de deploy gera
(`<equip>_inferencia.csv`: datetime, reconstruction_error, is_anomaly, severity).

Uso:
    python scripts/monitor_drift.py --inferencia <equip>_inferencia.csv --alarm <bundle>/alarm.json \
        [--dados <csv bruto> --bundle <pasta do bundle>] [--png saida.png] [--csv saida.csv]
    python scripts/monitor_drift.py --make-drift-ref --dados <csv bruto> --bundle <pasta do bundle>

Dentro do pacote de deploy roda como `python3 monitor_drift.py ...`: aquele arquivo é GERADO a partir
deste módulo por `scripts/package_monitor.py` (com o detector embutido) e só depende de
pandas/numpy/scipy (+ matplotlib se --png).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

from abc import ABC, abstractmethod
from scipy import stats


# ── detector embutido (cópia GERADA de src/transpetro_modelos/drift/detectors.py) ──

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
        sampling: str = "random",
        *,
        samples: list[np.ndarray] | None = None,
        d_crit: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.k_consecutive = k_consecutive
        # None = k janelas CONSECUTIVAS; n = k janelas acima dentre as últimas n (tolera dia intercalado)
        self.persistence_window = persistence_window
        # "random": amostra sorteada da referência (o limite muda um pouco com a semente e o tamanho);
        # "quantile": n_reference quantis igualmente espaçados da referência inteira (determinístico)
        if sampling not in ("random", "quantile"):
            raise ValueError("sampling deve ser 'random' ou 'quantile'.")
        self.sampling = sampling

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
                if sampling == "quantile":
                    sample = np.quantile(col, np.linspace(0, 1, min(len(col), n_reference)))
                else:
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
            sampling=drift_ref.get("sampling", "random"),
            samples=[drift_ref["sensors"][c]["sample"] for c in names],
            d_crit=[drift_ref["sensors"][c]["d_crit"] for c in names],
        )

    def to_drift_ref(self, reference_window: list[str] | None = None, n_keep: int = 2000) -> dict:
        """Serializa no formato do drift_ref.json dos bundles."""
        rng = np.random.default_rng(0)
        sensors = {}
        for name, sample, dc in zip(self.feature_names, self.samples, self.d_crit):
            if len(sample) <= n_keep:
                keep = sample
            elif self.sampling == "quantile":
                keep = np.quantile(sample, np.linspace(0, 1, n_keep))
            else:
                keep = rng.choice(sample, size=n_keep, replace=False)
            sensors[name] = {"d_crit": float(dc), "sample": [round(float(v), 4) for v in keep]}
        out = {"reference_window": reference_window, "win": self.window_size,
               "k_consec": self.k_consecutive, "sampling": self.sampling, "sensors": sensors}
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

    def scan(self, frame: pd.DataFrame | np.ndarray, index=None, reset_gap=None) -> tuple[pd.DataFrame, list]:
        """Passa uma série inteira pelo detector, como o monitor faz em produção.

        `reset_gap` (ex.: "3D"): com índice temporal, um buraco maior que isso entre duas amostras
        (parada longa) zera a persistência, para dias de antes da parada não somarem com os de depois.

        Retorna (d_diario, disparos):
          d_diario : DataFrame com o D de cada janela avaliada, por feature, indexado pelo
                     último instante da janela (colunas = feature_names)
          disparos : lista [(instante, [features])]; após cada disparo o detector é resetado
                     e continua monitorando
        """
        if isinstance(frame, pd.DataFrame):
            index = frame.index if index is None else index
            X = frame[self.feature_names].to_numpy(dtype=float)
        else:
            X = np.asarray(frame, dtype=float)
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            index = pd.RangeIndex(len(X)) if index is None else index
        self.reset()
        gap = pd.Timedelta(reset_gap) if reset_gap is not None else None
        linhas, instantes, disparos = [], [], []
        anterior = None
        for t, x in zip(index, X):
            if gap is not None and anterior is not None and (t - anterior) > gap:
                self.reset()
            anterior = t
            fired = self.update(x)
            if not self._buffer:            # uma janela acabou de ser avaliada
                linhas.append(self.last_d.copy()); instantes.append(t)
            if fired:
                disparos.append((t, list(self.last_drift_features)))
                self.reset()
        d_diario = pd.DataFrame(linhas, index=pd.Index(instantes, name=getattr(index, "name", None)),
                                columns=self.feature_names)
        return d_diario, disparos


# ── detector do M8 embutido (cópia GERADA de src/transpetro_modelos/drift/residuo.py) ──

class ResidualLevelDetector:
    """Mediana diária do resíduo de um modelo de comportamento normal fora da faixa da referência."""

    def __init__(self, target: str, predictors: list[str], q: float = 0.005, k: int = 3, n: int = 5,
                 min_coverage: float = 0.5, reset_gap: str = "3D"):
        self.target, self.predictors = target, list(predictors)
        self.q, self.k, self.n = q, k, n
        self.min_coverage = min_coverage      # fração mínima do dia com operação para o dia contar
        self.reset_gap = pd.Timedelta(reset_gap)
        self.coef: np.ndarray | None = None
        self.intercept: float | None = None
        self.band: tuple[float, float] | None = None
        self.r2: float | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.predictors].to_numpy(dtype=float) @ self.coef + self.intercept

    def residuals(self, df: pd.DataFrame) -> pd.Series:
        """Resíduo instante a instante (medido − previsto)."""
        return df[self.target] - self.predict(df)

    def daily(self, df: pd.DataFrame) -> pd.Series:
        """Mediana diária do resíduo, só nos dias com operação suficiente."""
        r = self.residuals(df)
        passo = df.index.to_series().diff().median()
        minimo = self.min_coverage * (pd.Timedelta("1D") / passo)
        n = r.resample("1D").count()
        return r.resample("1D").median()[n >= minimo]

    def fit(self, reference: pd.DataFrame) -> "ResidualLevelDetector":
        ref = reference[self.predictors + [self.target]].dropna()
        X = np.column_stack([ref[self.predictors].to_numpy(dtype=float), np.ones(len(ref))])
        y = ref[self.target].to_numpy(dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]                  # mínimos quadrados = regressão linear
        self.coef, self.intercept = beta[:-1], float(beta[-1])
        res = y - X @ beta
        self.r2 = float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())
        lo, hi = self.daily(ref).quantile([self.q, 1 - self.q])
        self.band = (float(lo), float(hi))
        return self

    def to_dict(self) -> dict:
        return {"target": self.target, "predictors": self.predictors, "coef": [float(v) for v in self.coef],
                "intercept": self.intercept, "band": list(self.band), "r2": self.r2, "q": self.q, "k": self.k,
                "n": self.n, "min_coverage": self.min_coverage, "reset_gap": str(self.reset_gap)}

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualLevelDetector":
        det = cls(d["target"], d["predictors"], q=d["q"], k=d["k"], n=d["n"], min_coverage=d["min_coverage"],
                  reset_gap=d["reset_gap"])
        det.coef, det.intercept = np.asarray(d["coef"], dtype=float), float(d["intercept"])
        det.band, det.r2 = tuple(d["band"]), d.get("r2")
        return det

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


# ── Gatilhos da política (docs/politica_retreino.md, seção 3): regra k-de-n sobre as últimas n semanas
#    válidas (robusta a uma semana quieta no meio de um drift). Calibrados no B-8802B:
#    modelo saudável 2025-26: alarme semanal máx. 1,0 % (2 de 84 semanas > 0,5 %); erro mediano ~1× (2× em 2026);
#    modelo com drift (2022 lendo 2025-26): alarme mediano 7 %/semana, erro mediano 3,4×, 65 de 84 semanas > 2 %.
RULES = {   # metric: (limiar, k, n)  → dispara se a métrica passou do limiar em >= k das últimas n semanas válidas
    "amarelo": {"alarme_pct": (0.5, 2, 4), "erro_p50_rel": (2.0, 6, 8), "fora_clip_pct": (10.0, 2, 4),
                "congelado_h": (0.0, 1, 1)},
    "vermelho": {"alarme_pct": (2.0, 4, 6), "erro_p50_rel": (2.5, 6, 8), "fora_clip_pct": (25.0, 4, 6)},
}
# fora_clip_pct (M5) = maior fração semanal, entre os sensores, de instantes FORA da faixa de clip do treino
# (ali o valor é truncado e o modelo não vê o sensor → drift "por omissão", invisível ao alarme).
# Só é calculado com --dados (CSV bruto) + --bundle; sem eles a métrica fica ausente e as regras dela não se aplicam.
# congelado_h (M7) = horas da semana com DADO CONGELADO (3+ sensores com o mesmo valor por 12 h+: falha de
# aquisição, o histórico repete o último valor). Qualquer hora na última semana → amarelo: avisar a instrumentação.
# Esses trechos também ficam fora do M6 (calibração e varredura). Exige --dados + --bundle.
MIN_SAMPLES_WEEK = 288   # >= 1 dia de operação (5 min) para a semana contar


def weekly_table(res: pd.DataFrame, mu_ref: float, freq: str = "W") -> pd.DataFrame:
    g = res.groupby(pd.Grouper(freq=freq))
    w = pd.DataFrame({
        "n_instantes": g.size(),
        "alarme_pct": 100 * g["is_anomaly"].mean(),
        "atencao_pct": 100 * g["severity"].apply(lambda s: (s != "normal").mean()),
        "erro_p50_rel": g["reconstruction_error"].median() / mu_ref,
        "erro_p90_rel": g["reconstruction_error"].quantile(0.9) / mu_ref,
    })
    w["valida"] = w["n_instantes"] >= MIN_SAMPLES_WEEK
    return w


def k_of_n(flags: pd.Series, k: int, n: int) -> tuple[bool, int]:
    tail = flags.tail(n); return (int(tail.sum()) >= k) and (len(tail) >= k), int(tail.sum())


def status_at(wv: pd.DataFrame) -> tuple[str, list[str]]:
    status, reasons = "verde", []
    for level in ("amarelo", "vermelho"):
        for metric, (thr, k, n) in RULES[level].items():
            if metric not in wv.columns or wv[metric].isna().all(): continue
            fired, cnt = k_of_n(wv[metric] > thr, k, n)
            if fired:
                status = level; reasons.append(f"{metric} > {thr} em {cnt} das últimas {n} semanas (regra: ≥ {k})")
    return status, reasons


def evaluate(w: pd.DataFrame) -> dict:
    wv = w[w["valida"]]
    status, reasons = status_at(wv)
    timeline = pd.Series([status_at(wv.iloc[: i + 1])[0] for i in range(len(wv))], index=wv.index, name="status")
    return {"status": status, "reasons": reasons, "timeline": timeline,
            "ultima_semana": str(wv.index[-1].date()) if len(wv) else None,
            "alarme_pct_ultimas_4": float(wv["alarme_pct"].tail(4).mean()) if len(wv) else None,
            "erro_p50_rel_ultimas_4": float(wv["erro_p50_rel"].tail(4).mean()) if len(wv) else None,
            "semanas_validas": int(len(wv))}


def clip_saturation(dados_csv: Path, bundle_dir: Path, freq: str) -> pd.DataFrame:
    """M5: por semana, fração de instantes fora da faixa [lo, hi] de clip do treino, por sensor, e o máximo entre eles.
    Reusa o simpred_inference.py do pacote de deploy (pasta Transpetro/, 2 níveis acima do bundle) para aplicar
    exatamente os passos temporais do pipeline.json (filtro de operação, resample, transientes, seleção)."""
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    steps = json.loads((bundle_dir / "pipeline.json").read_text()); bounds = json.loads((bundle_dir / "clip_bounds.json").read_text())
    df = si.carregar_dados(dados_csv)
    for st in steps:
        if st["step"] in si._STEPS_TEMPORAIS: df = si._STEPS_TEMPORAIS[st["step"]](df, **{k: v for k, v in st.items() if k != "step"})
    out = pd.DataFrame(index=df.index)
    for c, (lo, hi) in bounds.items():
        if c in df.columns: out[f"fora_clip__{c}"] = ((df[c] < lo) | (df[c] > hi)).astype(float)
    wk = 100 * out.groupby(pd.Grouper(freq=freq)).mean()
    wk["fora_clip_pct"] = wk.max(axis=1); wk["fora_clip_sensor"] = wk.drop(columns="fora_clip_pct").idxmax(axis=1).str.replace("fora_clip__", "")
    return wk




# ═══ M6 — detector rápido de drift: KS diário por sensor (CalibratedKSDetector) ═══
# Calibração (--make-drift-ref) grava drift_ref.json DENTRO do bundle: amostra de referência por sensor
# + limiar do estatístico D calibrado na própria referência. Dispara quando 3 dos últimos 5 dias têm
# algum sensor acima do limiar. B-8802B: ~4 dias de atraso no drift real (3,4–8,4 d conforme o
# alinhamento das janelas), 0 falsos no controle sem drift com referência de 12 meses.
M6_WIN, M6_KCONSEC, M6_NWIN, M6_NREF = 288, 3, 5, 2000
M6_RESET_GAP = "3D"   # parada maior que isso zera a persistência (dias de antes não somam com os de depois)


def _temporal_steps(bundle_dir: Path, df, todos_sensores: bool = False):
    """Passos temporais do pipeline.json do bundle; `todos_sensores` pula o select_features (o M8 usa sensores
    que o modelo não usa, como as temperaturas do motor)."""
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    for st in json.loads((bundle_dir / "pipeline.json").read_text()):
        if todos_sensores and st["step"] == "select_features":
            continue
        if st["step"] in si._STEPS_TEMPORAIS:
            df = si._STEPS_TEMPORAIS[st["step"]](df, **{k: v for k, v in st.items() if k != "step"})
    return df


def _sem_congelado(bundle_dir: Path, df):
    """Tira os trechos de dado congelado (M7). Pacote de deploy antigo, sem a função: devolve df inalterado."""
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    return si.remove_frozen_segments(df) if hasattr(si, "remove_frozen_segments") else df


def frozen_weekly(dados_csv: Path, bundle_dir: Path, freq: str) -> pd.DataFrame:
    """M7: horas por semana com dado congelado, sobre os sensores que o modelo usa."""
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    if not hasattr(si, "frozen_mask"):
        return pd.DataFrame()
    m = si.frozen_mask(_temporal_steps(bundle_dir, si.carregar_dados(dados_csv)))
    return pd.DataFrame({"congelado_h": m.astype(float).groupby(pd.Grouper(freq=freq)).sum() * 5 / 60})


def make_drift_ref(dados_csv: Path, bundle_dir: Path, ref_start=None, ref_end=None) -> Path:
    """Calibra o M6 na janela normal do treino (alarm.json) e grava drift_ref.json no bundle."""
    alarm = json.loads((bundle_dir / "alarm.json").read_text())
    nw = alarm.get("threshold_calibration", {}).get("normal_window", {})
    ref_start = ref_start or nw.get("start"); ref_end = ref_end or nw.get("end")
    if not (ref_start and ref_end):
        raise SystemExit("bundle sem threshold_calibration.normal_window — passe --ref-start/--ref-end")
    df = _sem_congelado(bundle_dir, _temporal_steps(bundle_dir, si_carregar(bundle_dir, dados_csv)))
    ref = df[(df.index >= pd.Timestamp(ref_start)) & (df.index <= pd.Timestamp(ref_end))]
    # quantis da referência inteira (não um sorteio): o limite e a amostra guardada não dependem de semente
    det = CalibratedKSDetector(ref, window_size=M6_WIN, k_consecutive=M6_KCONSEC, persistence_window=M6_NWIN,
                               n_reference=M6_NREF, sampling="quantile")
    out = det.to_drift_ref(reference_window=[str(ref_start), str(ref_end)], n_keep=M6_NREF)
    path = bundle_dir / "drift_ref.json"; path.write_text(json.dumps(out))
    print(f"drift_ref.json gravado em {path}  (referência {ref_start} → {ref_end}, {len(ref)} obs, {len(ref.columns)} sensores)")
    return path


def ks_daily(dados_csv: Path, bundle_dir: Path) -> tuple[pd.DataFrame, list]:
    """M6 sobre toda a série: (D diário por sensor, disparos [(instante, [sensores])])."""
    det = CalibratedKSDetector.from_drift_ref(bundle_dir / "drift_ref.json")
    df = _sem_congelado(bundle_dir, _temporal_steps(bundle_dir, si_carregar(bundle_dir, dados_csv)))
    return det.scan(df, reset_gap=M6_RESET_GAP)


def ks_daily_fires(dados_csv: Path, bundle_dir: Path) -> list:
    """Só os disparos do M6: [(instante, [sensores])]."""
    return ks_daily(dados_csv, bundle_dir)[1]


# ═══ M8 — nível de temperatura de mancal descontada a estação (ResidualLevelDetector, drift/residuo.py) ═══
# O KS bruto compara um dia com o ano inteiro; em temperatura, a estação deixa o limite perto de 1 e um degrau no meio
# do ano passa batido (B-8802B, mancal LA, 28/03/2026). O M8 monitora o resíduo de uma regressão da temperatura contra
# sensores que carregam estação e carga. Calibração em residual_ref.json no bundle (--make-residual-ref).
# B-8802B: dispara 4 dias depois do degrau, 0 disparos em 2025. B-4064A: mudança pós-reparo em ~2,5 dias, 0 falsos.


def make_residual_ref(dados_csv: Path, bundle_dir: Path, alvos: list[str], preditores: list[str],
                      ref_start=None, ref_end=None) -> Path:
    """Calibra o M8 (um detector por alvo) na janela normal do treino e grava residual_ref.json no bundle."""
    alarm = json.loads((bundle_dir / "alarm.json").read_text())
    nw = alarm.get("threshold_calibration", {}).get("normal_window", {})
    ref_start = ref_start or nw.get("start"); ref_end = ref_end or nw.get("end")
    if not (ref_start and ref_end):
        raise SystemExit("bundle sem threshold_calibration.normal_window — passe --ref-start/--ref-end")
    df = _sem_congelado(bundle_dir, _temporal_steps(bundle_dir, si_carregar(bundle_dir, dados_csv), todos_sensores=True))
    ref = df[(df.index >= pd.Timestamp(ref_start)) & (df.index <= pd.Timestamp(ref_end))]
    dets = []
    for alvo in alvos:
        det = ResidualLevelDetector(alvo, preditores).fit(ref)
        print(f"M8 {alvo}: R² {det.r2:.2f}, faixa normal do resíduo {det.band[0]:.1f} a {det.band[1]:.1f}"
              + ("   [aviso: R² < 0,5, a regressão explica pouco o sensor; prefira deixá-lo só no M6]" if det.r2 < 0.5 else ""))
        dets.append(det.to_dict())
    path = bundle_dir / "residual_ref.json"
    path.write_text(json.dumps({"reference_window": [str(ref_start), str(ref_end)], "detectors": dets}, indent=1, ensure_ascii=False))
    print(f"residual_ref.json gravado em {path}  (referência {ref_start} → {ref_end}, {len(ref)} obs)")
    return path


def residual_fires(dados_csv: Path, bundle_dir: Path) -> list:
    """Disparos do M8 na série: [(dia, [sensor])]."""
    cfg = json.loads((bundle_dir / "residual_ref.json").read_text())
    df = _sem_congelado(bundle_dir, _temporal_steps(bundle_dir, si_carregar(bundle_dir, dados_csv), todos_sensores=True))
    fires = []
    for d in cfg["detectors"]:
        _, disp = ResidualLevelDetector.from_dict(d).scan(df)
        fires += [(t, [d["target"]]) for t in disp]
    return sorted(fires)


def si_carregar(bundle_dir: Path, dados_csv: Path):
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    return si.carregar_dados(dados_csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inferencia", default=None, help="CSV gerado pelo script de deploy (<equip>_inferencia.csv)")
    ap.add_argument("--alarm", default=None, help="alarm.json do bundle (fonte de mean_normal = μ do treino)")
    ap.add_argument("--freq", default="W", help="frequência de agregação pandas (default W = semanal)")
    ap.add_argument("--csv", default=None, help="salvar a tabela semanal neste CSV")
    ap.add_argument("--png", default=None, help="salvar figura (alarme %% e erro relativo por semana)")
    ap.add_argument("--ultimas", type=int, default=8, help="quantas semanas imprimir (default 8)")
    ap.add_argument("--dados", default=None, help="CSV BRUTO de entrada do deploy (habilita M5 = saturação do clip, M6 = KS diário e M7 = dado congelado)")
    ap.add_argument("--bundle", default=None, help="pasta do bundle (pipeline.json + clip_bounds.json); usa o simpred_inference.py do pacote")
    ap.add_argument("--make-drift-ref", action="store_true", help="só calibra e grava drift_ref.json no bundle (usa --dados + --bundle) e sai")
    ap.add_argument("--make-residual-ref", action="store_true",
                    help="só calibra e grava residual_ref.json (M8) no bundle (usa --dados + --bundle + --alvos + --preditores) e sai")
    ap.add_argument("--alvos", default=None, help="M8: temperaturas a monitorar, separadas por vírgula (ex.: 'Temperatura Bomba LA')")
    ap.add_argument("--preditores", default=None, help="M8: sensores que explicam o alvo (estação e carga), separados por vírgula")
    ap.add_argument("--ref-start", default=None); ap.add_argument("--ref-end", default=None)
    args = ap.parse_args()

    if args.make_drift_ref:
        make_drift_ref(Path(args.dados), Path(args.bundle), args.ref_start, args.ref_end); return 0
    if args.make_residual_ref:
        if not (args.alvos and args.preditores):
            raise SystemExit("--make-residual-ref precisa de --alvos e --preditores")
        make_residual_ref(Path(args.dados), Path(args.bundle), [a.strip() for a in args.alvos.split(",")],
                          [p.strip() for p in args.preditores.split(",")], args.ref_start, args.ref_end); return 0
    if not (args.inferencia and args.alarm):
        ap.error("--inferencia e --alarm são obrigatórios para monitorar")

    res = pd.read_csv(args.inferencia, index_col=0, parse_dates=True).sort_index()
    res["is_anomaly"] = res["is_anomaly"].astype(str).str.lower().isin(["true", "1"])
    alarm = json.load(open(args.alarm))
    cal = alarm.get("threshold_calibration") or {}
    mu_ref = cal.get("mean_normal")
    if mu_ref is None:   # bundle antigo sem calibração sigma: usa a mediana do 1º mês como referência (aviso)
        mu_ref = float(res["reconstruction_error"].iloc[: 30 * 288].median())
        print(f"[aviso] alarm.json sem threshold_calibration.mean_normal — usando mediana do 1º mês ({mu_ref:.4f}) como μ de referência")

    w = weekly_table(res, mu_ref, args.freq)
    m6_fires, m8_fires = [], []
    if args.dados and args.bundle:
        w = w.join(clip_saturation(Path(args.dados), Path(args.bundle), args.freq))
        w = w.join(frozen_weekly(Path(args.dados), Path(args.bundle), args.freq))
        if (Path(args.bundle) / "drift_ref.json").exists():
            m6_fires = ks_daily_fires(Path(args.dados), Path(args.bundle))
        if (Path(args.bundle) / "residual_ref.json").exists():
            m8_fires = residual_fires(Path(args.dados), Path(args.bundle))
    ev = evaluate(w)
    # M6: disparo de KS nas últimas 4 semanas eleva a pelo menos AMARELO (investigar)
    recentes = [f for f in m6_fires if f[0] >= res.index.max() - pd.Timedelta(days=28)]
    if recentes and ev["status"] == "verde":
        ev["status"] = "amarelo"
    for t, sens in recentes:
        ev["reasons"].append(f"M6 (KS diário): drift detectado via {', '.join(sens)} em {t:%d/%m/%Y}")
    # M8: nível de temperatura fora do normal (estação descontada) nas últimas 4 semanas → pelo menos AMARELO
    recentes8 = [f for f in m8_fires if f[0] >= res.index.max() - pd.Timedelta(days=28)]
    if recentes8 and ev["status"] == "verde":
        ev["status"] = "amarelo"
    for sens in sorted({s for _, ss in recentes8 for s in ss}):
        dias = [t for t, ss in m8_fires if sens in ss]
        primeiro = next(t for t in reversed(dias) if not any(pd.Timedelta(0) < t - u <= pd.Timedelta(days=28) for u in dias))
        ev["reasons"].append(f"M8 (resíduo): nível de {sens} fora do normal, estação descontada; disparos repetidos desde {primeiro:%d/%m/%Y}")

    pd.set_option("display.width", 160)
    print(f"Equipamento: {Path(args.inferencia).stem}   μ treino = {mu_ref:.4f}   limiar alarme = {alarm['threshold']:.4f}")
    if alarm.get("provisional"):
        pv = alarm["provisional"]
        print(f"[aviso] bundle PROVISÓRIO: {pv.get('months')} mês(es) de dado do normal novo (até {str(pv.get('train_end'))[:10]}); "
              "alertas com ressalva; próximos modelos com 3, 6 e 12 meses de dado")
    print(f"Período: {res.index.min()} → {res.index.max()}   semanas válidas: {ev['semanas_validas']}\n")
    show = [c for c in ("n_instantes", "alarme_pct", "atencao_pct", "erro_p50_rel", "erro_p90_rel", "fora_clip_pct", "fora_clip_sensor", "congelado_h") if c in w.columns]
    print(w[w["valida"]][show].tail(args.ultimas).round(3).to_string())
    if m6_fires:
        print(f"\nM6 (KS diário) — {len(m6_fires)} disparo(s) na série:")
        for t, sens in m6_fires[-5:]: print(f"    {t:%d/%m/%Y %H:%M}  via {', '.join(sens)}")
    elif args.dados and args.bundle and (Path(args.bundle) / "drift_ref.json").exists():
        print("\nM6 (KS diário): nenhum disparo na série ✓")
    if m8_fires:
        print(f"\nM8 (nível no resíduo) — {len(m8_fires)} disparo(s) na série; o 1º e os últimos:")
        for t, sens in ([m8_fires[0]] + m8_fires[-4:] if len(m8_fires) > 5 else m8_fires): print(f"    {t:%d/%m/%Y}  {', '.join(sens)}")
    elif args.dados and args.bundle and (Path(args.bundle) / "residual_ref.json").exists():
        print("\nM8 (nível no resíduo): nenhum disparo na série ✓")
    cor = {"verde": "VERDE — operação normal, nada a fazer", "amarelo": "AMARELO — investigar com a operação (manutenção? regime novo? sensor?); NÃO retreinar ainda",
           "vermelho": "VERMELHO — drift sustentado: aplicar checklist drift × degradação e, confirmado, abrir retreino"}
    print(f"\n>>> STATUS: {cor[ev['status']]}")
    for r in ev["reasons"]: print(f"    - {r}")
    print(f"    últimas 4 semanas: alarme {ev['alarme_pct_ultimas_4']:.2f} %  ·  erro p50 relativo {ev['erro_p50_rel_ultimas_4']:.2f}×")
    tl = ev["timeline"]; vc = tl.value_counts()
    print(f"    histórico do semáforo ({len(tl)} semanas): " + "  ".join(f"{k}={int(vc.get(k, 0))}" for k in ("verde", "amarelo", "vermelho")))
    first_red = tl[tl == "vermelho"].index.min(); first_amb = tl[tl == "amarelo"].index.min()
    if pd.notna(first_amb): print(f"    1º amarelo: {first_amb.date()}" + (f"   1º vermelho: {first_red.date()}" if pd.notna(first_red) else ""))
    if args.csv: tl.to_frame().join(w).to_csv(args.csv)

    if args.png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        wv = w[w["valida"]]
        has5 = "fora_clip_pct" in wv.columns
        fig, ax = plt.subplots(3 if has5 else 2, 1, figsize=(12, 8 if has5 else 5.5), sharex=True)
        ax[0].bar(wv.index, wv["alarme_pct"], width=6, color="#2a78d6"); ax[0].axhline(RULES["amarelo"]["alarme_pct"][0], color="#eda100", ls="--", lw=1, label="amarelo 0,5 %")
        ax[0].axhline(RULES["vermelho"]["alarme_pct"][0], color="#e34948", ls="--", lw=1, label="vermelho 2 %"); ax[0].set_ylabel("alarme na semana (%)"); ax[0].legend(fontsize=8, frameon=False)
        ax[1].plot(wv.index, wv["erro_p50_rel"], color="#2a78d6", lw=1.5); ax[1].axhline(1, color="#52514e", lw=.8); ax[1].axhline(2, color="#eda100", ls="--", lw=1); ax[1].axhline(2.5, color="#e34948", ls="--", lw=1)
        ax[1].set_ylabel("erro mediano ÷ μ treino")
        if has5:
            ax[2].bar(wv.index, wv["fora_clip_pct"], width=6, color="#2a78d6"); ax[2].axhline(10, color="#eda100", ls="--", lw=1); ax[2].axhline(25, color="#e34948", ls="--", lw=1)
            ax[2].set_ylabel("fora da faixa de clip (%)\n(pior sensor)")
        fig.suptitle(f"Monitor de drift — {Path(args.inferencia).stem} — status {ev['status'].upper()}", x=0.01, ha="left", fontsize=11)
        cmap = {"verde": "#1baf7a", "amarelo": "#eda100", "vermelho": "#e34948"}
        for t, st in ev["timeline"].items():
            if st != "verde": ax[0].axvspan(t - pd.Timedelta(days=7), t, color=cmap[st], alpha=.18, lw=0)
        for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", lw=.5, color="#e6e5e1"); a.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(args.png, dpi=130); print(f"figura: {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
