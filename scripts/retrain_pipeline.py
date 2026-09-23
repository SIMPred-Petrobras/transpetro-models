"""
Pipeline de retreino (estágio 3 da política de drift) — executa a receita validada no B-8802B-2025:
  check   → portão: janela aprovada pela operação + dados mínimos (>= 12 meses e >= 4000 h de operação)
  train   → grade local compacta (seeds × hiperparâmetros), seleção por FP held-out (régua de deploy μ+6,5σ + 15/20)
  package → bundle de deploy autocontido (pesos + arch + scaler + clip + pipeline + alarm + drift_ref)
  battery → bateria completa (scripts/battery.py) no bundle empacotado; reprova → tenta o próximo candidato

NUNCA roda sem `--operacao-confirmou` (registro de que a operação confirmou que a janela é operação normal).

Ex. (replay do retreino do B-8802B):
  python scripts/retrain_pipeline.py --equipment B-8802B-2025 --train-start 2025-01-01 --train-end 2026-01-01 \
      --heldout-end 2026-06-01 --out results/replay/retreino --operacao-confirmou
"""
import argparse, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training.automl import build_model
from transpetro_modelos.training.train import train_vae
from transpetro_modelos.training.evaluate import compute_vae_errors

_spec = importlib.util.spec_from_file_location("battery", ROOT / "scripts/battery.py")
battery = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(battery)
_spec2 = importlib.util.spec_from_file_location("monitor_drift", ROOT / "scripts/monitor_drift.py")
mon = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(mon)

GRID = [  # (layers, latent, lr) — em volta da região vencedora das buscas anteriores; 3 seeds cada
    ((128, 64, 32), 16, 1e-4), ((128, 64, 32), 16, 1e-3), ((64, 32, 16), 16, 1e-3), ((256, 128, 64), 16, 1e-3),
]
SEEDS = (0, 1, 2)
Y_ALARM, Y_ATT, K_P, N_P = 6.5, 4.0, 15, 20


def kofn(f, k=K_P, n=N_P): return (f.astype(int).rolling(n, min_periods=n).sum() >= k).fillna(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipment", required=True)
    ap.add_argument("--train-start", required=True); ap.add_argument("--train-end", required=True)
    ap.add_argument("--heldout-end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--operacao-confirmou", action="store_true",
                    help="registro do portão humano: operação confirmou que a janela de treino é operação normal")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--max-candidates", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = EQUIPMENT_CONFIGS[a.equipment]
    ts, te, he = pd.Timestamp(a.train_start), pd.Timestamp(a.train_end), pd.Timestamp(a.heldout_end)

    # ── check (portão) ──────────────────────────────────────────────────────────
    if not a.operacao_confirmou:
        raise SystemExit("PORTÃO: rode com --operacao-confirmou após a operação validar a janela (política, seção 4).")
    raw = load_equipment_data(a.equipment, from_clearml=False)
    pre, _, _ = run_preprocessing(raw, cfg.pre_split_steps)
    tr_idx = pre[(pre.index >= ts) & (pre.index < te)]
    horas = len(tr_idx) / 12; meses = (te - ts).days / 30.4
    print(f"[check] janela {ts.date()} → {te.date()}: {meses:.1f} meses, {horas:.0f} h de operação", flush=True)
    if meses < 11.5 or horas < 4000:
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
    t0 = pd.Timestamp(sy["t0"]); tf = t0 + pd.Timedelta(hours=sy["ramp_h"])
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
            fp_ho = 100 * fl[(fl.index >= te) & (fl.index < he)].mean()
            fp_po = 100 * fl[fl.index >= he].mean()
            eI = pd.Series(compute_vae_errors(m, XI), index=XI.index)
            fI = kofn(eI > mu + Y_ALARM * sd); w_ = fI[(fI.index >= t0)]
            fi = w_[w_].index.min() if w_.any() else None
            synth_h = (tf - fi).total_seconds() / 3600 if fi is not None else -999.0
            rank.append({"tag": tag, "model": m, "mu": mu, "sd": sd, "fp_ho": fp_ho, "fp_po": fp_po,
                         "synth_h": synth_h, "layers": layers, "latent": latent})
            print(f"[train] {tag:36s} FP held-out {fp_ho:.3f}%  posterior {fp_po:.3f}%  sintética {synth_h:.0f} h", flush=True)
    # ranking: entre os que cabem no teto de FP, maximiza sensibilidade; desempate por FP
    fp_max = battery.BATTERY[a.equipment]["criteria"]["max_fp_heldout_pct"]
    rank.sort(key=lambda r: (r["fp_ho"] > fp_max, -r["synth_h"], r["fp_ho"], r["fp_po"]))

    # ── package + battery: empacota o melhor e valida; reprovou → próximo ───────
    data_csv = battery.BATTERY[a.equipment]["data_new"]
    for i, cand in enumerate(rank[: a.max_candidates], 1):
        bdir = out / f"model_{ts.date()}_{te.date()}_VAE"
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
            "features": list(full.columns),
            "threshold_calibration": {"method": "sigma", "mean_normal": float(cand["mu"]), "std_normal": float(cand["sd"]),
                                      "y_alarm": Y_ALARM, "y_attention": Y_ATT,
                                      "persistence": {"k": K_P, "n": N_P},
                                      "normal_window": {"start": str(ts), "end": str(te)},
                                      "candidate": cand["tag"]}}, indent=1, ensure_ascii=False))
        print(f"\n[package] candidato #{i} ({cand['tag']}) → {bdir}", flush=True)
        mon.make_drift_ref(Path(data_csv), bdir)
        print(f"[battery] candidato #{i}:", flush=True)
        res = battery.run_battery(a.equipment, bundle=bdir, train_end=str(te.date()), heldout_end=str(he.date()))
        (out / f"battery_cand{i}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        if res["approved"]:
            print(f"\n>>> PIPELINE CONCLUÍDO: bundle aprovado em {bdir}\n    Próximo passo: 4 semanas em SOMBRA ao lado do bundle atual antes da troca (política, seção 5).")
            return 0
        print(f"[battery] candidato #{i} REPROVADO — tentando o próximo…\n", flush=True)
    print(">>> PIPELINE INTERROMPIDO: nenhum candidato aprovado pela bateria. Manter o bundle atual e investigar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
