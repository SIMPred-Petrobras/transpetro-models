"""
Pipeline de retreino (estágio 3 da política de drift) — executa a receita validada no B-8802B-2025:
  check   → portão: janela aprovada pela operação + dados mínimos (>= 12 meses e >= 4000 h de operação)
  train   → grade local compacta (seeds × hiperparâmetros), seleção por FP held-out (régua de deploy μ+6,5σ + 15/20)
  package → bundle de deploy autocontido (pesos + arch + scaler + clip + pipeline + alarm + drift_ref + residual_ref)
  battery → bateria completa (scripts/battery.py) no bundle empacotado; reprova → tenta o próximo candidato

NUNCA roda sem `--operacao-confirmou` (registro de que a operação confirmou que a janela é operação normal).

Modelo PROVISÓRIO (`--provisorio`): depois de uma mudança confirmada, não se espera 12 meses com o modelo desatualizado.
Com >= 1 mês e >= 300 h do normal novo treina-se um provisório, refeito todo mês com tudo o que acumulou (mesmo
--train-start, --train-end avançando um mês), até virar o definitivo com 12 meses. Diferenças: FP medido na validação
(fim da janela; não há dado depois), falha sintética numa data escolhida no último mês da janela, bateria provisória
(critérios mais brandos) e o bundle marcado como provisório no alarm.json. No B-8802B o provisório quase não deu alarme
falso desde o 1º mês, mas só detectou falha de forma confiável com ~8 meses: os alertas dele valem com ressalva.
  python scripts/retrain_pipeline.py --equipment B-8802B-2025 --provisorio --train-start 2025-01-06 \
      --train-end 2025-02-06 --out results/provisorio/m01 --operacao-confirmou

Ex. (replay do retreino do B-8802B):
  python scripts/retrain_pipeline.py --equipment B-8802B-2025 --train-start 2025-01-01 --train-end 2026-01-01 \
      --heldout-end 2026-06-01 --out results/replay/retreino --operacao-confirmou
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training.automl import build_model
from transpetro_modelos.training.train import train_vae
from transpetro_modelos.training.evaluate import compute_vae_errors

from transpetro_modelos.drift import battery, monitor as mon

GRID = [  # (layers, latent, lr) — em volta da região vencedora das buscas anteriores; 3 seeds cada
    ((128, 64, 32), 16, 1e-4), ((128, 64, 32), 16, 1e-3), ((64, 32, 16), 16, 1e-3), ((256, 128, 64), 16, 1e-3),
]
SEEDS = (0, 1, 2)
Y_ALARM, Y_ATT, K_P, N_P = 6.5, 4.0, 15, 20


def kofn(f, k=K_P, n=N_P): return (f.astype(int).rolling(n, min_periods=n).sum() >= k).fillna(False)


def aplicar_rampa(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    """Degradação lenta simulada: `COLUNA:INICIO:DELTA_POR_MES` → coluna + delta × meses desde INICIO, só com a bomba
    operando (mesma regra de `battery.inject`). Usada no teste da janela deslizante (scripts/janela_deslizante.py)."""
    col, ini, delta = spec.rsplit(":", 2)
    out = df.copy()
    meses = np.clip(np.asarray((out.index - pd.Timestamp(ini)).total_seconds()) / (86400 * 30.4), 0, None)
    out[col] = out[col].values + float(delta) * meses * (out["Pressão Descarga"] > 35).values
    print(f"[rampa] {col}: +{float(delta)}/mês desde {ini} (até +{float(delta) * meses.max():.2f} no fim do dado)", flush=True)
    return out


def _t0_sintetica(raw: pd.DataFrame, cfg, ts, te, sy) -> pd.Timestamp:
    """Data da falha sintética do provisório: a mais tarde, no último mês da janela, com a bomba operando >= 80 %
    do tempo da injeção (rampa + platô)."""
    run = next((s for s in cfg.pre_split_steps if s["step"] == "filter_running"), None)
    dur = pd.Timedelta(hours=sy["ramp_h"] + sy["hold_h"])
    for k in range(0, 31):
        t0 = (te - dur - pd.Timedelta(days=k)).normalize()
        if t0 < ts:
            break
        w = raw.loc[t0: t0 + dur]
        if run is None or (len(w) and (w[run["column"]] > run["threshold"]).mean() >= 0.8):
            return t0
    raise SystemExit("não há janela com a bomba operando no último mês para injetar a falha sintética")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipment", required=True)
    ap.add_argument("--train-start", required=True); ap.add_argument("--train-end", required=True)
    ap.add_argument("--heldout-end", default=None, help="fim do held-out (modelo definitivo; o provisório não usa)")
    ap.add_argument("--provisorio", action="store_true",
                    help="modelo provisório: >= 1 mês do normal novo, FP na validação, bateria provisória")
    ap.add_argument("--out", required=True)
    ap.add_argument("--operacao-confirmou", action="store_true",
                    help="registro do portão humano: operação confirmou que a janela de treino é operação normal")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument("--from-clearml", action="store_true", help="lê o dado de treino do ClearML Dataset (worker remoto)")
    ap.add_argument("--rampa", default=None, metavar="COLUNA:INICIO:DELTA_POR_MES",
                    help="EXPERIMENTO: soma ao dado uma degradação lenta (rampa linear, só com a bomba operando) "
                         "para testar se o retreino a aprende como normal; ex. 'Vibração Bomba LNA:2026-01-06:0.15'")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = EQUIPMENT_CONFIGS[a.equipment]
    if not a.provisorio and not a.heldout_end:
        raise SystemExit("--heldout-end é obrigatório no modelo definitivo")
    ts, te = pd.Timestamp(a.train_start), pd.Timestamp(a.train_end)
    he = te if a.provisorio else pd.Timestamp(a.heldout_end)

    # ── check (portão) ──────────────────────────────────────────────────────────
    if not a.operacao_confirmou:
        raise SystemExit("PORTÃO: rode com --operacao-confirmou após a operação validar a janela (política, seção 4).")
    raw = load_equipment_data(a.equipment, from_clearml=a.from_clearml)
    if a.rampa:
        raw = aplicar_rampa(raw, a.rampa)
    pre, _, _ = run_preprocessing(raw, cfg.pre_split_steps)
    tr_idx = pre[(pre.index >= ts) & (pre.index < te)]
    horas = len(tr_idx) / 12; meses = (te - ts).days / 30.4
    print(f"[check] janela {ts.date()} → {te.date()}: {meses:.1f} meses, {horas:.0f} h de operação", flush=True)
    if a.provisorio:
        if meses < 0.9 or horas < 300:
            raise SystemExit(f"PORTÃO: janela insuficiente para o provisório (mínimos: 1 mês e 300 h; tem {meses:.1f} m / {horas:.0f} h).")
        if meses >= 11.5 and horas >= 4000:
            print("[check] a janela já tem 12 meses: rode o modelo DEFINITIVO (sem --provisorio)", flush=True)
    elif meses < 11.5 or horas < 4000:
        raise SystemExit(f"PORTÃO: janela insuficiente (mínimos: 12 meses e 4000 h; tem {meses:.1f} m / {horas:.0f} h).")

    # ── train: grade local, ranking por FP held-out com a régua de DEPLOY ──────
    preset = get_preprocessing_steps(a.equipment, "baseline")
    n_val = int(len(tr_idx) * 0.15); fit, va = tr_idx.iloc[:-n_val], tr_idx.iloc[-n_val:]
    Xtr, art, _ = run_preprocessing(fit, preset, return_artifacts=True, return_report=True)
    Xva = run_preprocessing(va, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    full = run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    dl = lambda X, sh, bs=256: DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float32)), batch_size=bs, shuffle=sh)
    # sensibilidade rápida no ranking (lição das buscas anteriores: menor FP sozinho seleciona o modelo mais CEGO):
    # injeta a falha sintética a 100% e mede a antecedência com a mesma régua — a bateria continua sendo a autoridade.
    sy = battery.BATTERY[a.equipment]["synthetic"]
    t0 = _t0_sintetica(raw, cfg, ts, te, sy) if a.provisorio else pd.Timestamp(sy["t0"])
    tf = t0 + pd.Timedelta(hours=sy["ramp_h"])
    print(f"[check] falha sintética injetada em {t0:%d/%m/%Y %H:%M}", flush=True)
    rawI = battery.inject(raw, sy["signature"], t0, sy["ramp_h"], sy["hold_h"], 1.0)
    preI, _, _ = run_preprocessing(rawI, cfg.pre_split_steps)
    XI = run_preprocessing(preI[(preI.index >= t0 - pd.Timedelta(days=2)) & (preI.index <= tf + pd.Timedelta(hours=sy["hold_h"] + 12))],
                           preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    rank = []; cache = out / "cands"; cache.mkdir(exist_ok=True)
    for layers, latent, lr in GRID:
        for seed in SEEDS:
            tag = f"l{'-'.join(map(str, layers))}_lat{latent}_lr{lr:g}_s{seed}"
            m = build_model("vae", Xtr.shape[1], dense_layers=list(layers), latent_dim=latent)
            if (cache / f"{tag}.pt").exists():
                m.load_state_dict(torch.load(cache / f"{tag}.pt", map_location="cpu")); m.eval()
            else:
                torch.manual_seed(seed); np.random.seed(seed)
                m = train_vae(m, dl(Xtr, True), dl(Xva, False), epochs=a.epochs, learning_rate=lr, weight_decay=1e-5, patience=10)
                m.eval(); torch.save(m.state_dict(), cache / f"{tag}.pt")
            torch.manual_seed(0)
            e = pd.Series(compute_vae_errors(m, full), index=full.index)
            mtr = e[(e.index >= ts) & (e.index < te)]; mu, sd = float(mtr.mean()), float(mtr.std())
            fl = kofn(e > mu + Y_ALARM * sd)
            if a.provisorio:   # sem dado depois do treino: FP na validação (fim da janela, fora dos pesos)
                fp_ho = 100 * fl[(fl.index >= va.index.min()) & (fl.index < te)].mean(); fp_po = 0.0
            else:
                fp_ho = 100 * fl[(fl.index >= te) & (fl.index < he)].mean()
                fp_po = 100 * fl[fl.index >= he].mean()
            eI = pd.Series(compute_vae_errors(m, XI), index=XI.index)
            fI = kofn(eI > mu + Y_ALARM * sd); w_ = fI[(fI.index >= t0)]
            fi = w_[w_].index.min() if w_.any() else None
            synth_h = (tf - fi).total_seconds() / 3600 if fi is not None else -999.0
            if fl[(fl.index >= t0) & (fl.index <= tf + pd.Timedelta(hours=sy["hold_h"]))].any():
                synth_h = -999.0   # já alarma ali sem a falha: um alarme "antecipado" não mede detecção
            rank.append({"tag": tag, "model": m, "mu": mu, "sd": sd, "fp_ho": fp_ho, "fp_po": fp_po,
                         "synth_h": synth_h, "layers": layers, "latent": latent})
            print(f"[train] {tag:36s} FP held-out {fp_ho:.3f}%  posterior {fp_po:.3f}%  sintética {synth_h:.0f} h", flush=True)
    # ranking: entre os que cabem no teto de FP, maximiza sensibilidade; desempate por FP
    fp_max = (battery.BATTERY[a.equipment]["criteria_provisional"]["max_fp_val_pct"] if a.provisorio
              else battery.BATTERY[a.equipment]["criteria"]["max_fp_heldout_pct"])
    rank.sort(key=lambda r: (r["fp_ho"] > fp_max, -r["synth_h"], r["fp_ho"], r["fp_po"]))

    # ── package + battery: empacota o melhor e valida; reprovou → próximo ───────
    data_csv = battery.BATTERY[a.equipment]["data_new"]
    for i, cand in enumerate(rank[: a.max_candidates], 1):
        bdir = out / (f"model_{ts.date()}_{te.date()}_VAE" + ("_provisorio" if a.provisorio else ""))
        bdir.mkdir(exist_ok=True)
        torch.save(cand["model"].state_dict(), bdir / "model_state.pt")
        (bdir / "model_arch.json").write_text(json.dumps({"model_type": "vae", "input_dim": full.shape[1],
                                                          "encoding_layers": list(cand["layers"]), "latent_dim": cand["latent"]}, indent=1))
        pickle.dump(art.scaler, open(bdir / "scaler.pkl", "wb"))
        (bdir / "clip_bounds.json").write_text(json.dumps({c: list(v) for c, v in art.clip_bounds.items()}, indent=1, ensure_ascii=False))
        (bdir / "pipeline.json").write_text(json.dumps(list(cfg.pre_split_steps) + preset, indent=1, ensure_ascii=False))
        (bdir / "alarm.json").write_text(json.dumps({
            "model_type": "vae", "threshold": float(cand["mu"] + Y_ALARM * cand["sd"]),
            "threshold_attention": float(cand["mu"] + Y_ATT * cand["sd"]),
            "debounce_consecutive": 6, "debounce_window": N_P, "debounce_min": K_P,
            "persistence_mode": "time", "persistence_step": "5min", "min_alert_hours": 1.0,
            "features": list(full.columns),
            "threshold_calibration": {"method": "sigma", "mean_normal": float(cand["mu"]), "std_normal": float(cand["sd"]),
                                      "y_alarm": Y_ALARM, "y_attention": Y_ATT,
                                      "persistence": {"k": K_P, "n": N_P},
                                      "normal_window": {"start": str(ts), "end": str(te)},
                                      "candidate": cand["tag"]},
            **({"provisional": {"months": round(meses, 1), "train_start": str(ts), "train_end": str(te),
                                "note": "modelo provisório: alertas com ressalva; no B-8802B a detecção de falha só ficou "
                                        "confiável com ~8 meses de dado. Retreinar todo mês até 12 meses (definitivo)."}}
               if a.provisorio else {})}, indent=1, ensure_ascii=False))
        print(f"\n[package] candidato #{i} ({cand['tag']}) → {bdir}", flush=True)
        mon.make_drift_ref(Path(data_csv), bdir)
        # M8: herda do bundle em produção quais temperaturas monitorar e com quais preditores, e recalibra no
        # período de treino do bundle novo (a referência dos detectores = o período em que o modelo aprendeu)
        rr_atual = Path(battery.BATTERY[a.equipment]["bundle"]) / "residual_ref.json"
        if rr_atual.exists():
            m8 = json.loads(rr_atual.read_text())["detectors"]
            mon.make_residual_ref(Path(data_csv), bdir, [d["target"] for d in m8], m8[0]["predictors"])
        print(f"[battery] candidato #{i}:", flush=True)
        if a.provisorio:
            res = battery.run_battery(a.equipment, bundle=bdir, train_end=str(te), provisional=True,
                                      val_start=str(va.index.min()), synthetic_t0=str(t0))
        else:
            res = battery.run_battery(a.equipment, bundle=bdir, train_end=str(te.date()), heldout_end=str(he.date()))
        (out / f"battery_cand{i}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        if res["approved"]:
            if a.provisorio:
                print(f"\n>>> PIPELINE CONCLUÍDO: bundle PROVISÓRIO aprovado em {bdir}\n    Alertas com ressalva. "
                      f"Próximo passo: sombra ao lado do bundle atual; daqui a 1 mês, rodar de novo com --train-end avançado.")
            else:
                print(f"\n>>> PIPELINE CONCLUÍDO: bundle aprovado em {bdir}\n    Próximo passo: 4 semanas em SOMBRA ao lado do bundle atual antes da troca (política, seção 5).")
            return 0
        print(f"[battery] candidato #{i} REPROVADO — tentando o próximo…\n", flush=True)
    print(">>> PIPELINE INTERROMPIDO: nenhum candidato aprovado pela bateria. Manter o bundle atual e investigar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
