"""
Monitor de drift do SIMPred — calcula, semana a semana, os indicadores da política de retreino
(docs/politica_retreino.md) a partir do CSV que o próprio pacote de deploy gera
(`<equip>_inferencia.csv`: datetime, reconstruction_error, is_anomaly, severity).

Uso:
    python scripts/monitor_drift.py --inferencia deploy_v2/Transpetro/B-8802B-2025/scripts/b8802b2025_inferencia.csv \
        --alarm deploy_v2/Transpetro/B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE/alarm.json [--png saida.png] [--csv saida.csv]

Só depende de pandas/numpy (+ matplotlib se --png). Não usa a lib interna: pode rodar no ambiente da integração.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

# ── Gatilhos da política (docs/politica_retreino.md, seção 3): regra k-de-n sobre as últimas n semanas
#    válidas (robusta a uma semana quieta no meio de um drift). Calibrados no B-8802B:
#    modelo saudável 2025-26: alarme semanal máx. 1,0 % (2 de 84 semanas > 0,5 %); erro mediano ~1× (2× em 2026);
#    modelo com drift (2022 lendo 2025-26): alarme mediano 7 %/semana, erro mediano 3,4×, 65 de 84 semanas > 2 %.
RULES = {   # metric: (limiar, k, n)  → dispara se a métrica passou do limiar em >= k das últimas n semanas válidas
    "amarelo": {"alarme_pct": (0.5, 2, 4), "erro_p50_rel": (2.0, 6, 8), "fora_clip_pct": (10.0, 2, 4)},
    "vermelho": {"alarme_pct": (2.0, 4, 6), "erro_p50_rel": (2.5, 6, 8), "fora_clip_pct": (25.0, 4, 6)},
}
# fora_clip_pct (M5) = maior fração semanal, entre os sensores, de instantes FORA da faixa de clip do treino
# (ali o valor é truncado e o modelo não vê o sensor → drift "por omissão", invisível ao alarme).
# Só é calculado com --dados (CSV bruto) + --bundle; sem eles a métrica fica ausente e as regras dela não se aplicam.
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




# ═══ M6 — detector rápido de drift: KS diário por sensor (univariado, calibrado no treino) ═══
# Calibração (--make-drift-ref) grava drift_ref.json DENTRO do bundle: amostra de referência por sensor
# + limiar do estatístico D auto-calibrado (máximo do D diário dentro da própria referência — absorve
# autocorrelação e regimes normais; validado: 2,6-4 d de atraso nos drifts rotulados, 0 falsos com
# referência de 12 meses). Dispara com K_CONSEC dias consecutivos acima do limiar em algum sensor.
M6_WIN, M6_KCONSEC, M6_NREF = 288, 3, 2000


def _temporal_steps(bundle_dir: Path, df):
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    for st in json.loads((bundle_dir / "pipeline.json").read_text()):
        if st["step"] in si._STEPS_TEMPORAIS:
            df = si._STEPS_TEMPORAIS[st["step"]](df, **{k: v for k, v in st.items() if k != "step"})
    return df


def make_drift_ref(dados_csv: Path, bundle_dir: Path, ref_start=None, ref_end=None) -> Path:
    from scipy.stats import ks_2samp
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    alarm = json.loads((bundle_dir / "alarm.json").read_text())
    nw = alarm.get("threshold_calibration", {}).get("normal_window", {})
    ref_start = ref_start or nw.get("start"); ref_end = ref_end or nw.get("end")
    if not (ref_start and ref_end):
        raise SystemExit("bundle sem threshold_calibration.normal_window — passe --ref-start/--ref-end")
    df = _temporal_steps(bundle_dir, si.carregar_dados(dados_csv))
    ref = df[(df.index >= pd.Timestamp(ref_start)) & (df.index <= pd.Timestamp(ref_end))]
    rng = np.random.default_rng(0)
    out = {"reference_window": [str(ref_start), str(ref_end)], "win": M6_WIN, "k_consec": M6_KCONSEC, "sensors": {}}
    for c in ref.columns:
        vals = ref[c].dropna().values
        sample = rng.choice(vals, size=min(len(vals), 5000), replace=False)
        dcrit = max(ks_2samp(sample, vals[i:i + M6_WIN]).statistic for i in range(0, max(1, len(vals) - M6_WIN), M6_WIN))
        keep = rng.choice(sample, size=min(len(sample), M6_NREF), replace=False)
        out["sensors"][c] = {"d_crit": float(dcrit), "sample": [round(float(v), 4) for v in keep]}
    path = bundle_dir / "drift_ref.json"; path.write_text(json.dumps(out))
    print(f"drift_ref.json gravado em {path}  (referência {ref_start} → {ref_end}, {len(ref)} obs, {len(ref.columns)} sensores)")
    return path


def ks_daily_fires(dados_csv: Path, bundle_dir: Path):
    """Roda o KS diário sobre toda a série e retorna a lista de disparos [(data, [sensores])]."""
    from scipy.stats import ks_2samp
    ref = json.loads((bundle_dir / "drift_ref.json").read_text())
    df = _temporal_steps(bundle_dir, si_carregar(bundle_dir, dados_csv))
    win, k = ref["win"], ref["k_consec"]
    samples = {c: np.asarray(v["sample"]) for c, v in ref["sensors"].items() if c in df.columns}
    dcrit = {c: ref["sensors"][c]["d_crit"] for c in samples}
    fires, run_above = [], []
    for i in range(0, len(df) - win, win):
        w = df.iloc[i:i + win]
        above = [c for c in samples if ks_2samp(samples[c], w[c].values).statistic > dcrit[c]]
        run_above = run_above + [set(above)] if above else []
        if len(run_above) >= k:
            comuns = set.intersection(*run_above[-k:]) or set().union(*run_above[-k:])
            fires.append((w.index[-1], sorted(comuns))); run_above = []
    return fires


def si_carregar(bundle_dir: Path, dados_csv: Path):
    sys.path.insert(0, str(bundle_dir.resolve().parents[2]))
    import simpred_inference as si
    return si.carregar_dados(dados_csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inferencia", required=True, help="CSV gerado pelo script de deploy (<equip>_inferencia.csv)")
    ap.add_argument("--alarm", required=True, help="alarm.json do bundle (fonte de mean_normal = μ do treino)")
    ap.add_argument("--freq", default="W", help="frequência de agregação pandas (default W = semanal)")
    ap.add_argument("--csv", default=None, help="salvar a tabela semanal neste CSV")
    ap.add_argument("--png", default=None, help="salvar figura (alarme %% e erro relativo por semana)")
    ap.add_argument("--ultimas", type=int, default=8, help="quantas semanas imprimir (default 8)")
    ap.add_argument("--dados", default=None, help="CSV BRUTO de entrada do deploy (habilita M5 = saturação do clip e M6 = KS diário)")
    ap.add_argument("--bundle", default=None, help="pasta do bundle (pipeline.json + clip_bounds.json); usa o simpred_inference.py do pacote")
    ap.add_argument("--make-drift-ref", action="store_true", help="só calibra e grava drift_ref.json no bundle (usa --dados + --bundle) e sai")
    ap.add_argument("--ref-start", default=None); ap.add_argument("--ref-end", default=None)
    args = ap.parse_args()

    if args.make_drift_ref:
        make_drift_ref(Path(args.dados), Path(args.bundle), args.ref_start, args.ref_end); return 0

    res = pd.read_csv(args.inferencia, index_col=0, parse_dates=True).sort_index()
    res["is_anomaly"] = res["is_anomaly"].astype(str).str.lower().isin(["true", "1"])
    alarm = json.load(open(args.alarm))
    cal = alarm.get("threshold_calibration") or {}
    mu_ref = cal.get("mean_normal")
    if mu_ref is None:   # bundle antigo sem calibração sigma: usa a mediana do 1º mês como referência (aviso)
        mu_ref = float(res["reconstruction_error"].iloc[: 30 * 288].median())
        print(f"[aviso] alarm.json sem threshold_calibration.mean_normal — usando mediana do 1º mês ({mu_ref:.4f}) como μ de referência")

    w = weekly_table(res, mu_ref, args.freq)
    m6_fires = []
    if args.dados and args.bundle:
        w = w.join(clip_saturation(Path(args.dados), Path(args.bundle), args.freq))
        if (Path(args.bundle) / "drift_ref.json").exists():
            m6_fires = ks_daily_fires(Path(args.dados), Path(args.bundle))
    ev = evaluate(w)
    # M6: disparo de KS nas últimas 4 semanas eleva a pelo menos AMARELO (investigar)
    recentes = [f for f in m6_fires if f[0] >= res.index.max() - pd.Timedelta(days=28)]
    if recentes and ev["status"] == "verde":
        ev["status"] = "amarelo"
    for t, sens in recentes:
        ev["reasons"].append(f"M6 (KS diário): drift detectado via {', '.join(sens)} em {t:%d/%m/%Y}")

    pd.set_option("display.width", 160)
    print(f"Equipamento: {Path(args.inferencia).stem}   μ treino = {mu_ref:.4f}   limiar alarme = {alarm['threshold']:.4f}")
    print(f"Período: {res.index.min()} → {res.index.max()}   semanas válidas: {ev['semanas_validas']}\n")
    show = [c for c in ("n_instantes", "alarme_pct", "atencao_pct", "erro_p50_rel", "erro_p90_rel", "fora_clip_pct", "fora_clip_sensor") if c in w.columns]
    print(w[w["valida"]][show].tail(args.ultimas).round(3).to_string())
    if m6_fires:
        print(f"\nM6 (KS diário) — {len(m6_fires)} disparo(s) na série:")
        for t, sens in m6_fires[-5:]: print(f"    {t:%d/%m/%Y %H:%M}  via {', '.join(sens)}")
    elif args.dados and args.bundle and (Path(args.bundle) / "drift_ref.json").exists():
        print("\nM6 (KS diário): nenhum disparo na série ✓")
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
