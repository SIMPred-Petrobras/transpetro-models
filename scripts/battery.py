"""
Bateria de validação de um bundle de deploy (paridade total com produção: usa o simpred_inference.py
do pacote, os limiares e a persistência do alarm.json do próprio bundle).

Testes (por equipamento, declarados em BATTERY):
  1. FP na janela de treino, no held-out temporal e no período posterior
  2. Episódios-âncora do período recente continuam alarmando (apagá-los = modelo cego, não melhor)
  3. Cross-era: falha real histórica (série que o modelo não viu) — antecedência mínima e FP no normal da época
  4. Falha sintética: assinatura real injetada (rampa+platô) em 100% e 50% da intensidade

Uso:
  python scripts/battery.py --equipment B-8802B-2025 [--bundle <dir>] [--json saida.json]
Sai com código 0 se TODOS os critérios passam; 1 caso contrário.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEP = ROOT / "deploy_v2/Transpetro"
sys.path.insert(0, str(DEP))
import simpred_inference as si  # noqa: E402

BATTERY = {
    "B-8802B-2025": {
        "bundle": DEP / "B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE",
        "data_new": DEP / "B-8802B-2025/dados/2025_2026/data_2025-01-01_2026-08-10_raw.csv",
        "train_end": "2026-01-01", "heldout_end": "2026-06-01",
        "anchors": {  # episódios de prioridade ALTA (docs/justificativa_alarmes_b8802b_2025.md); tolerância ±2 h
            "ep2 17/01/26 vib LA+LNA": ("2026-01-17 21:30", "2026-01-17 22:40"),
            "ep4 10/08/26 sucção/vib": ("2026-08-10 12:45", "2026-08-10 16:25"),
        },
        "cross_era": {
            "csv": DEP / "B-8802B/dados", "csv_glob": "*_raw.csv",
            "failure": "2022-07-06 10:00", "ramp_start": "2022-06-29",
            "min_lead_days": 2.0, "max_normal_rate_pct": 0.5,
        },
        "synthetic": {  # assinatura medida da rampa de 2022 (ramp − normal)
            "signature": {"Pressão Sucção": -0.059, "Pressão Descarga": -1.535, "Vibração Bomba LA": 1.515,
                          "Vibração Bomba LNA": 1.237, "Temperatura Bomba LA": 6.699},
            "t0": "2026-06-10", "ramp_h": 48, "hold_h": 24,
            "min_lead_h_100": 20.0, "require_50_detected": True,
        },
        "criteria": {"max_fp_heldout_pct": 0.05, "max_fp_post_pct": 0.30},
    },
}


def score(bundle: Path, df: pd.DataFrame) -> pd.DataFrame:
    return si.prever(bundle, si.carregar_modelo(bundle), si.preprocessar(bundle, df))


def inject(df: pd.DataFrame, sig: dict, t0: pd.Timestamp, ramp_h: int, hold_h: int, scale: float) -> pd.DataFrame:
    inj = df.copy()
    dt = np.asarray((inj.index - t0).total_seconds()) / 3600.0
    prof = np.clip(dt / ramp_h, 0, 1); prof = np.where(dt < 0, 0, prof); prof = np.where(dt > ramp_h + hold_h, 0, prof)
    run = (inj["Pressão Descarga"] > 35).values
    for c, dv in sig.items():
        if c in inj.columns: inj[c] = inj[c].values + scale * dv * prof * run
    return inj


def run_battery(equipment: str, bundle: Path | None = None, data_new: Path | None = None,
                train_end: str | None = None, heldout_end: str | None = None, verbose=True) -> dict:
    cfg = BATTERY[equipment]
    bundle = Path(bundle or cfg["bundle"]); data_new = Path(data_new or cfg["data_new"])
    train_end = pd.Timestamp(train_end or cfg["train_end"]); heldout_end = pd.Timestamp(heldout_end or cfg["heldout_end"])
    P = lambda *a: print(*a, flush=True) if verbose else None
    P(f"Bateria — {equipment}\n  bundle: {bundle}\n")

    raw = si.carregar_dados(data_new)
    res = score(bundle, raw)
    fl = res["is_anomaly"]
    tr, ho, post = fl[fl.index < train_end], fl[(fl.index >= train_end) & (fl.index < heldout_end)], fl[fl.index >= heldout_end]
    out = {"fp_treino_pct": 100 * tr.mean(), "fp_heldout_pct": 100 * ho.mean(), "fp_post_pct": 100 * post.mean()}
    P(f"[1] FP: treino {out['fp_treino_pct']:.3f}%  held-out {out['fp_heldout_pct']:.3f}%  posterior {out['fp_post_pct']:.3f}%")

    out["anchors"] = {}
    for nome, (a, b) in cfg["anchors"].items():
        w = fl[(fl.index >= pd.Timestamp(a) - pd.Timedelta(hours=2)) & (fl.index <= pd.Timestamp(b) + pd.Timedelta(hours=2))]
        out["anchors"][nome] = bool(w.any())
        P(f"[2] âncora {nome}: {'mantida ✓' if w.any() else 'PERDIDA ✗'}")

    ce = cfg["cross_era"]
    raw_old = si.carregar_dados(next(Path(ce["csv"]).rglob(ce["csv_glob"])))
    r_old = score(bundle, raw_old); f_old = r_old["is_anomaly"]
    failure, ramp0 = pd.Timestamp(ce["failure"]), pd.Timestamp(ce["ramp_start"])
    ramp = f_old[(f_old.index >= ramp0) & (f_old.index < failure)]
    first = ramp[ramp].index.min() if ramp.any() else None
    out["cross_era"] = {"lead_days": (failure - first).total_seconds() / 86400 if first is not None else None,
                        "normal_rate_pct": 100 * f_old[f_old.index < ramp0].mean(),
                        "first_alarm": str(first)}
    P(f"[3] cross-era: 1º alarme {first}  antecedência {out['cross_era']['lead_days'] and round(out['cross_era']['lead_days'],2)} d  "
      f"normal pré-rampa {out['cross_era']['normal_rate_pct']:.2f}%")

    sy = cfg["synthetic"]; t0 = pd.Timestamp(sy["t0"]); tf = t0 + pd.Timedelta(hours=sy["ramp_h"])
    out["synthetic"] = {}
    for s in (1.0, 0.5):
        rI = score(bundle, inject(raw, sy["signature"], t0, sy["ramp_h"], sy["hold_h"], s))
        w = rI.loc[t0: tf + pd.Timedelta(hours=sy["hold_h"]), "is_anomaly"]
        fi = w[w].index.min() if w.any() else None
        lead = (tf - fi).total_seconds() / 3600 if fi is not None else None
        out["synthetic"][f"{int(s*100)}"] = lead
        P(f"[4] sintética {int(s*100)}%: {'não detecta' if lead is None else f'{lead:.1f} h antes do fim da rampa'}")

    cr = cfg["criteria"]
    checks = {
        f"FP held-out ≤ {cr['max_fp_heldout_pct']}%": out["fp_heldout_pct"] <= cr["max_fp_heldout_pct"],
        f"FP posterior ≤ {cr['max_fp_post_pct']}%": out["fp_post_pct"] <= cr["max_fp_post_pct"],
        "âncoras mantidas": all(out["anchors"].values()),
        f"cross-era ≥ {ce['min_lead_days']} d e normal ≤ {ce['max_normal_rate_pct']}%":
            out["cross_era"]["lead_days"] is not None and out["cross_era"]["lead_days"] >= ce["min_lead_days"]
            and out["cross_era"]["normal_rate_pct"] <= ce["max_normal_rate_pct"],
        f"sintética 100% ≥ {sy['min_lead_h_100']} h": out["synthetic"]["100"] is not None and out["synthetic"]["100"] >= sy["min_lead_h_100"],
        "sintética 50% detectada": (out["synthetic"]["50"] is not None) or not sy["require_50_detected"],
    }
    out["checks"] = checks; out["approved"] = all(checks.values())
    P("\nCritérios:")
    for k, v in checks.items(): P(f"  {'✓' if v else '✗'} {k}")
    P(f"\n>>> {'APROVADO' if out['approved'] else 'REPROVADO'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipment", required=True, choices=list(BATTERY))
    ap.add_argument("--bundle", default=None); ap.add_argument("--data-new", default=None)
    ap.add_argument("--train-end", default=None); ap.add_argument("--heldout-end", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = run_battery(a.equipment, a.bundle, a.data_new, a.train_end, a.heldout_end)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0 if out["approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
