"""
Exploração de engenharia de features e de subconjuntos de sensores no modelo de anomalia do B-8802B (out/2026).

Pergunta: dá para deixar o erro de reconstrução mais estável (menos oscilação e menos efeito da estação na temperatura)
sem perder a detecção? O diagnóstico do modelo de produção mostra que, em abr–ago/2026, o erro dobra e a parcela da
temperatura do mancal LA fica ~9× maior (o mancal esfriou para ~45 °C, abaixo do que o treino de 2025 viu).

Variantes (todas com o mesmo filtro de operação do bundle, treino 2025, mesma arquitetura e régua de alarme):
  base         5 sensores, como o bundle de produção (controle)
  vib          só as 2 vibrações
  vib_press    vibrações + pressões (sem temperatura)
  temp_delta   vib_press + variação da temperatura do mancal LA em 30 min e 2 h (T(t) − T(t−Δ)), sem o nível
  temp_rel     vib_press + temperatura do mancal LA menos a sua mediana dos últimos 3 dias de operação
  temps        só as 4 temperaturas (mancais da bomba e do motor)
  espectro     base + desvio-padrão de 1 h e energia por faixa de período (FFT em janelas de 2 h) das vibrações.
               Os dados são médias de 5 min: o "espectro" vê oscilações de 10 min a 2 h, não a frequência mecânica.

Cada variante: 2 taxas de aprendizado × 3 sementes. Cada candidato é avaliado com o score bruto e com o score
suavizado (mediana móvel causal de 1 h); o limiar é μ+6,5σ do score em 2025 nos dois casos, seguido da persistência
do deploy (15 de 20 leituras em 100 min, só com medição real) e da regra de 1 h.

Métricas: alerta em 2025 / jan–mai/26 / jun–ago/26, episódios em 2026, falha real de 2022 (antecedência) e alerta no
normal de 2022 (outro conceito, informativo), falha sintética em 5 datas de 2026 a 100% e 50% (assinatura de 2022 nos
8 sensores), rugosidade do score (desvio da variação de 5 min ÷ desvio do score) e parcela do erro por entrada.

    OMP_NUM_THREADS=4 nice -n 19 python scripts/fe_exploracao_b8802b.py       # ~1–2 h na CPU; retoma do cache
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transpetro_modelos.drift import battery  # noqa: E402
from transpetro_modelos.training.automl import build_model  # noqa: E402
from transpetro_modelos.training.train import train_vae  # noqa: E402

si = battery.si
CFG = battery.BATTERY["B-8802B-2025"]
BUNDLE = CFG["bundle"]
TS, TE, HE = map(pd.Timestamp, ("2025-01-01", "2026-01-01", "2026-06-01"))
DATAS = ["2026-01-25", "2026-03-05", "2026-04-20", "2026-06-10", "2026-07-15"]
RAMP, HOLD = 48, 24
SIG = dict(CFG["synthetic"]["signature"],
           **{"Temperatura Bomba LNA": -1.7, "Temperatura Motor LA": 2.6, "Temperatura Motor LNA": 1.8})
CE = CFG["cross_era"]
FAIL, R0 = pd.Timestamp(CE["failure"]), pd.Timestamp(CE["ramp_start"])
GRADE = [((128, 64, 32), 16, 1e-4), ((128, 64, 32), 16, 1e-3)]
SEEDS = (0, 1, 2)
Y_ALARM = 6.5
SUAVE = 12                      # mediana móvel de 12 leituras de 5 min = 1 h

PS, PD_, VLA, VLNA, TLA = "Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA", "Vibração Bomba LNA", "Temperatura Bomba LA"
TEMPS = [TLA, "Temperatura Bomba LNA", "Temperatura Motor LA", "Temperatura Motor LNA"]


# ── pré-processamento temporal (o mesmo do bundle, sem escolher colunas) ──────────
def temporal(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for s in json.loads((BUNDLE / "pipeline.json").read_text()):
        if s["step"] in si._STEPS_TEMPORAIS and s["step"] != "select_features":
            df = si._STEPS_TEMPORAIS[s["step"]](df, **{k: v for k, v in s.items() if k != "step"})
    return df


def segmentos(idx: pd.DatetimeIndex) -> np.ndarray:
    """Trechos contínuos de operação (5 min sem buraco): janelas curtas reiniciam em cada parada."""
    return idx.to_series().diff().ne(pd.Timedelta("5min")).cumsum().to_numpy()


def por_trecho(s: pd.Series, f) -> pd.Series:
    return s.groupby(segmentos(s.index)).transform(f)


def espectro(x: pd.Series, n: int = 24) -> pd.DataFrame:
    """Energia (log) por faixa de período numa janela de n leituras (2 h), atribuída ao fim da janela."""
    faixas = {"60-120min": (1, 2), "20-40min": (3, 6), "10-17min": (7, 12)}
    out = pd.DataFrame(np.nan, index=x.index, columns=list(faixas))
    seg = segmentos(x.index)
    for g in np.unique(seg):
        pos = np.flatnonzero(seg == g)
        if len(pos) < n:
            continue
        w = sliding_window_view(x.to_numpy()[pos], n)
        p = np.abs(np.fft.rfft(w - w.mean(axis=1, keepdims=True), axis=1)) ** 2
        for nome, (a, b) in faixas.items():
            out.iloc[pos[n - 1:], out.columns.get_loc(nome)] = np.log1p(p[:, a:b + 1].sum(axis=1))
    return out


def features(pre: pd.DataFrame, variante: str) -> pd.DataFrame:
    vp = pre[[PS, PD_, VLA, VLNA]]
    if variante == "base":
        return pre[[PS, PD_, VLA, VLNA, TLA]]
    if variante == "vib":
        return pre[[VLA, VLNA]]
    if variante == "vib_press":
        return vp
    if variante == "temp_delta":
        t = pre[TLA]
        return vp.assign(**{"ΔT 30 min": por_trecho(t, lambda s: s - s.shift(6)),
                            "ΔT 2 h": por_trecho(t, lambda s: s - s.shift(24))})
    if variante == "temp_rel":
        t = pre[TLA]
        return vp.assign(**{"T − mediana 3 d": t - t.rolling(864, min_periods=72).median()})
    if variante == "temps":
        return pre[TEMPS]
    if variante == "espectro":
        extra = []
        for c, nome in [(VLA, "LA"), (VLNA, "LNA")]:
            e = espectro(pre[c]).add_prefix(f"vib {nome} ")
            e[f"vib {nome} dp 1 h"] = por_trecho(pre[c], lambda s: s.rolling(12, min_periods=3).std())
            extra.append(e)
        return pd.concat([pre[[PS, PD_, VLA, VLNA, TLA]]] + extra, axis=1)
    raise ValueError(variante)


VARIANTES = ["base", "vib", "vib_press", "temp_delta", "temp_rel", "temps", "espectro"]


class Escala:
    """Clip [p1, p99,9] + escala robusta ajustados no treino; NaN (início de trecho) vira a mediana (0)."""
    def fit(self, X: pd.DataFrame):
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.999)
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median()
        self.iqr = (Xc.quantile(0.75) - Xc.quantile(0.25)).replace(0, 1.0)
        return self

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        return ((X.clip(self.lo, self.hi, axis=1) - self.med) / self.iqr).fillna(0.0)


def erros(m, X: pd.DataFrame) -> pd.DataFrame:
    """Erro quadrático por entrada (a média das colunas é o erro de reconstrução do deploy)."""
    m.eval()                    # em eval o VAE usa a média do espaço latente (sem amostrar)
    with torch.no_grad():
        x = torch.tensor(X.values, dtype=torch.float32)
        r = m(x)[0]
        return pd.DataFrame(((r - x) ** 2).numpy(), index=X.index, columns=X.columns)


def suavizar(score: pd.Series) -> pd.Series:
    return score.groupby(segmentos(score.index)).transform(lambda s: s.rolling(SUAVE, min_periods=1).median())


def alertas(score: pd.Series, thr: float, raw: pd.DataFrame) -> pd.Series:
    real = si._instantes_com_operacao(BUNDLE, raw, score.index)
    a = (score > thr)[real].astype(float)
    acima = ((a.rolling("100min").sum() >= 15) & (a.rolling("100min").count() >= 20)).reindex(score.index, fill_value=False)
    return si._duracao_minima(acima, 1.0)


def episodios(f: pd.Series) -> int:
    x = f[f]
    return 0 if x.empty else int((x.index.to_series().diff() > pd.Timedelta("1h")).sum() + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/fe_exploracao")
    ap.add_argument("--variantes", nargs="*", default=VARIANTES)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    out = ROOT / a.out
    (out / "scores").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    raw = si.carregar_dados(CFG["data_new"])
    raw22 = si.carregar_dados(next(Path(CE["csv"]).rglob(CE["csv_glob"])))
    pre, pre22 = temporal(raw), temporal(raw22)
    print(f"dado 2025-26: {len(pre)} leituras de 5 min com a bomba operando; 2022: {len(pre22)}", flush=True)

    arq = out / "avaliacao.csv"
    linhas = pd.read_csv(arq).to_dict("records") if arq.exists() else []
    feitos = {(r["variante"], r["cand"]) for r in linhas}
    janelas = {d: raw.loc[pd.Timestamp(d) - pd.Timedelta(days=5): pd.Timestamp(d) + pd.Timedelta(hours=RAMP + HOLD + 24)]
               for d in DATAS}
    liga = (raw[PD_] > 35).astype(float).resample("5min").max()
    operando = {d: liga.loc[pd.Timestamp(d): pd.Timestamp(d) + pd.Timedelta(hours=RAMP + HOLD)].mean() for d in DATAS}
    print("fração do tempo com a bomba operando em cada falha sintética:", {d: round(v, 2) for d, v in operando.items()})

    for var in a.variantes:
        F = features(pre, var)
        tr = F[(F.index >= TS) & (F.index < TE)]
        n_val = int(len(tr) * 0.15)
        esc = Escala().fit(tr.iloc[:-n_val])
        X, X22 = esc(F), esc(features(pre22, var))
        Xfit, Xva = X.loc[tr.index[:-n_val]], X.loc[tr.index[-n_val:]]
        dl = lambda Z, sh: DataLoader(TensorDataset(torch.tensor(Z.values, dtype=torch.float32)), batch_size=256, shuffle=sh)
        cache = out / "cands" / var
        cache.mkdir(parents=True, exist_ok=True)
        for layers, latent, lr in GRADE:
            for seed in SEEDS:
                tag = f"l{'-'.join(map(str, layers))}_lat{latent}_lr{lr:g}_s{seed}"
                if (var, tag) in feitos:
                    continue
                t_ini = time.time()
                m = build_model("vae", X.shape[1], dense_layers=list(layers), latent_dim=latent)
                if (cache / f"{tag}.pt").exists():
                    m.load_state_dict(torch.load(cache / f"{tag}.pt", map_location="cpu"))
                else:
                    torch.manual_seed(seed); np.random.seed(seed)
                    m = train_vae(m, dl(Xfit, True), dl(Xva, False), epochs=a.epochs, learning_rate=lr,
                                  weight_decay=1e-5, patience=10)
                    torch.save(m.state_dict(), cache / f"{tag}.pt")
                m.eval()
                E, E22 = erros(m, X), erros(m, X22)
                e, e22 = E.mean(axis=1), E22.mean(axis=1)
                parcela = (E.loc["2026-04-01":].sum() / E.loc["2026-04-01":].values.sum() * 100).round(1)
                pd.DataFrame({"score": e.astype("float32")}).to_parquet(out / "scores" / f"{var}__{tag}.parquet")
                for modo in ("bruto", "suave"):
                    s, s22 = (e, e22) if modo == "bruto" else (suavizar(e), suavizar(e22))
                    st = s[(s.index >= TS) & (s.index < TE)]
                    thr = float(st.mean() + Y_ALARM * st.std())
                    f, f22 = alertas(s, thr, raw), alertas(s22, thr, raw22)
                    rp = f22[(f22.index >= R0) & (f22.index < FAIL)]
                    first = rp[rp].index.min() if rp.any() else None
                    lin = {"variante": var, "cand": tag, "modo": modo, "n_entradas": X.shape[1],
                           "rugosidade": float(s.diff().std() / s.std()),
                           "erro_2026T2T3/2025": float(s.loc["2026-04-01":].mean() / st.mean()),
                           "alerta_2025_%": 100 * f[f.index < TE].mean(),
                           "alerta_jan-mai26_%": 100 * f[(f.index >= TE) & (f.index < HE)].mean(),
                           "alerta_jun-ago26_%": 100 * f[f.index >= HE].mean(),
                           "episodios_2026": episodios(f[f.index >= TE]),
                           "falha2022_dias": (FAIL - first).total_seconds() / 86400 if first is not None else np.nan,
                           "normal2022_%": 100 * f22[f22.index < R0].mean(),
                           "parcela_abr-ago26": json.dumps(parcela.to_dict(), ensure_ascii=False)}
                    for d in DATAS:
                        t0 = pd.Timestamp(d); tf = t0 + pd.Timedelta(hours=RAMP); fim = tf + pd.Timedelta(hours=HOLD)
                        for esc_pct in (100, 50):
                            col = f"sint{esc_pct}_{d[5:]}_h"
                            if f[(f.index >= t0) & (f.index <= fim)].any() or operando[d] < 0.8:
                                lin[col] = np.nan; continue     # já alerta sem a falha, ou bomba parada: não mede
                            rj = battery.inject(janelas[d], SIG, t0, RAMP, HOLD, esc_pct / 100)
                            Xj = esc(features(temporal(rj), var))
                            ej = erros(m, Xj).mean(axis=1)
                            ej = ej if modo == "bruto" else suavizar(ej)
                            w = alertas(ej, thr, rj).loc[t0:fim]
                            fi = w[w].index.min() if w.any() else None
                            lin[col] = round((tf - fi).total_seconds() / 3600, 1) if fi is not None else -99.0
                    linhas.append(lin)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lin.items() if k != "parcela_abr-ago26"},
                          flush=True)
                pd.DataFrame(linhas).to_csv(arq, index=False)
                print(f"[{var} {tag}] {time.time() - t_ini:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
