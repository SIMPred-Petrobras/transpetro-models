"""
Relatório de investigação de drift — gerado automaticamente quando o monitor sai do verde.
Preenche o checklist drift × degradação da política (docs/politica_retreino.md) com o que é computável
e sugere uma pré-conclusão; a DECISÃO continua humana.

Uso:
  python scripts/drift_report.py --inferencia <csv> --alarm <alarm.json> --dados <csv bruto> \
      --bundle <dir> --out <pasta> [--until AAAA-MM-DD]   # --until: simula o relatório naquela data
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

from transpetro_modelos.drift import monitor as mon


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inferencia", required=True); ap.add_argument("--alarm", required=True)
    ap.add_argument("--dados", required=True); ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--until", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle); alarm = json.loads(Path(a.alarm).read_text())
    mu_ref = alarm.get("threshold_calibration", {}).get("mean_normal")

    res = pd.read_csv(a.inferencia, index_col=0, parse_dates=True).sort_index()
    res["is_anomaly"] = res["is_anomaly"].astype(str).str.lower().isin(["true", "1"])
    raw = mon.si_carregar(bundle, Path(a.dados))
    if a.until:
        cut = pd.Timestamp(a.until); res, raw = res[res.index <= cut], raw[raw.index <= cut]
    hoje = res.index.max()
    if mu_ref is None: mu_ref = float(res["reconstruction_error"].iloc[: 30 * 288].median())

    # semáforo + M5 + M6 (mesmas funções do monitor, sobre o dado cortado)
    tmp = Path(tempfile.mkdtemp()); raw.to_csv(tmp / "raw.csv")
    w = mon.weekly_table(res, mu_ref).join(mon.clip_saturation(tmp / "raw.csv", bundle, "W"))
    ev = mon.evaluate(w)
    fires = mon.ks_daily_fires(tmp / "raw.csv", bundle) if (bundle / "drift_ref.json").exists() else []
    rec_fires = [f for f in fires if f[0] >= hoje - pd.Timedelta(days=28)]
    if rec_fires and ev["status"] == "verde": ev["status"] = "amarelo"

    # início estimado da mudança = 1ª semana não-verde ou 1º disparo de M6
    tl = ev["timeline"]; nv = tl[tl != "verde"]
    inicio = min([t for t in [nv.index.min() if len(nv) else None, fires[0][0] if fires else None] if t is not None], default=None)

    # ── checklist computável ─────────────────────────────────────────────────────
    feats = mon._temporal_steps(bundle, raw)
    dia = feats.resample("D").median().dropna(how="all")
    from scipy.stats import theilslopes
    trend = {}
    for c in [c for c in feats.columns if "Vibra" in c or "Temperatura" in c]:
        y = dia[c].dropna().tail(30)
        if len(y) < 15: continue
        slope, *_ = theilslopes(y.values, np.arange(len(y)))
        subindo = float((y.diff() > 0).mean())
        trend[c] = {"slope_dia": float(slope), "consistencia": subindo,
                    "monotônica": bool(abs(slope) > 0.05 and (subindo > 0.65 or subindo < 0.35))}
    err_d = res["reconstruction_error"].resample("D").median()
    corr_regime = float(err_d.corr(dia["Pressão Descarga"])) if "Pressão Descarga" in dia else np.nan
    wv = w[w["valida"]]
    clip4 = wv["fora_clip_pct"].tail(4).mean() if "fora_clip_pct" in wv else np.nan
    al = res["is_anomaly"]; ep = al[al]
    n_ep = 0
    if len(ep): n_ep = 1 + int((pd.Series(ep.index).diff() > pd.Timedelta(hours=12)).sum())
    horas_op = int(len(feats[feats.index >= inicio]) / 12) if inicio is not None else 0

    deg = any(v["monotônica"] and v["slope_dia"] > 0 for v in trend.values())
    if deg and ev["status"] != "verde":
        pre = "POSSÍVEL DEGRADAÇÃO — há tendência monotônica de subida em sensor físico: tratar como ALARME à operação; NÃO retreinar."
    elif ev["status"] != "verde":
        pre = "DRIFT PROVÁVEL — desvio sem tendência monotônica de degradação (padrão de regime/patamar). Confirmar com a operação (manutenção? mudança operacional? sensor?)."
    else:
        pre = "Sem sinal acionável no momento."

    L = [f"# Relatório de investigação de drift — {bundle.parents[1].name}",
         f"\nGerado em (dados até): **{hoje}** · bundle `{bundle.name}` · política: `docs/politica_retreino.md`",
         f"\n## Status do semáforo: **{ev['status'].upper()}**"]
    L += [f"- {r}" for r in ev["reasons"]] or ["- (verde)"]
    if rec_fires: L += ["\n### M6 — KS diário (últimas 4 semanas)"] + [f"- {t:%d/%m/%Y}: drift via **{', '.join(s)}**" for t, s in rec_fires]
    L += [f"\nInício estimado da mudança: **{inicio}**  ·  horas de operação desde então: **{horas_op} h** (mínimo p/ retreino: 4000 h e 12 meses de janela)"]
    L += ["\n## Checklist drift × degradação (pré-preenchido)",
          "\n**1. Tendência monotônica em vibração/temperatura (30 d, mediana diária)?**"]
    for c, v in trend.items():
        L += [f"- {c}: {v['slope_dia']:+.3f}/dia, {100*v['consistencia']:.0f}% dos dias subindo → {'⚠️ MONOTÔNICA' if v['monotônica'] else 'não'}"]
    L += [f"\n**2. Erro acompanha o regime (corr. diária erro × P. descarga, tudo até a data)?** {corr_regime:+.2f} "
          f"({'sim — padrão de drift de regime' if abs(corr_regime) > 0.3 else 'fraca'})",
          f"\n**3. Saturação do clip (M5, últimas 4 semanas):** {clip4:.1f} % ({'⚠️ modelo cego a parte da faixa' if clip4 == clip4 and clip4 > 10 else 'ok'})",
          f"\n**4. Alarmes:** {int(al.sum())} instantes em {n_ep} episódio(s) — {'intermitente (padrão de drift)' if n_ep >= 3 else 'concentrado'}",
          "\n**5. Perguntas à operação (responder antes de decidir):** houve manutenção/intervenção? mudança de faixa operacional? sensor trocado/recalibrado? o período pós-mudança foi operação normal?",
          f"\n## Pré-conclusão sugerida\n> {pre}",
          "\n## Saídas possíveis (política, seção 4)",
          "- **Degradação** → alarme à operação; NÃO retreinar.",
          "- **Drift leve** (erro relativo ≤ 1,5×) → recalibrar régua (`scripts/recalibrate_threshold.py`).",
          f"- **Drift confirmado** → retreino (`scripts/retrain_pipeline.py`), exigindo janela aprovada pela operação e dados mínimos ({'ATINGIDOS' if horas_op >= 4000 else f'AINDA NÃO: {horas_op}/4000 h'})."]
    # figura
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].bar(wv.index, wv["alarme_pct"], width=6, color="#2a78d6"); ax[0].axhline(0.5, color="#eda100", ls="--", lw=1); ax[0].axhline(2, color="#e34948", ls="--", lw=1); ax[0].set_ylabel("alarme (%)")
    ax[1].plot(wv.index, wv["erro_p50_rel"], color="#2a78d6", lw=1.4); ax[1].axhline(2, color="#eda100", ls="--", lw=1); ax[1].axhline(2.5, color="#e34948", ls="--", lw=1); ax[1].set_ylabel("erro ÷ μ treino")
    if "fora_clip_pct" in wv: ax[2].bar(wv.index, wv["fora_clip_pct"], width=6, color="#2a78d6"); ax[2].axhline(10, color="#eda100", ls="--", lw=1); ax[2].axhline(25, color="#e34948", ls="--", lw=1); ax[2].set_ylabel("fora do clip (%)")
    for t, _ in fires:
        for a_ in ax: a_.axvline(t, color="#e34948", lw=1, ls=":")
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False); a_.grid(axis="y", lw=.5, color="#e6e5e1"); a_.set_axisbelow(True)
    fig.suptitle(f"Monitor até {hoje:%d/%m/%Y} — status {ev['status'].upper()} (pontilhado vermelho = disparos M6)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "monitor.png", dpi=130)
    L += ["\n![monitor](monitor.png)"]
    (out / "relatorio_drift.md").write_text("\n".join(L))
    print(f"relatório: {out / 'relatorio_drift.md'}  (status {ev['status'].upper()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
