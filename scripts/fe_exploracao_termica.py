"""Complemento de fe_exploracao_b8802b.py: o que cada variante perde numa falha SÓ TÉRMICA (temperaturas sobem, vibração
e pressões não mudam). Usa os modelos já treinados em results/fe_exploracao/cands. Rampa de 48 h + platô de 24 h com a
parte térmica da assinatura de 2022 (mancal LA +6,7 °C) a 1× e 2× (+13,4 °C), nas 3 datas de 2026 com a bomba operando.

    python scripts/fe_exploracao_termica.py      # ~5 min
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_exploracao_b8802b as fx  # noqa: E402

DATAS = ["2026-01-25", "2026-06-10", "2026-07-15"]
TERMICA = {k: v for k, v in fx.SIG.items() if "Temperatura" in k}


def main():
    out = fx.ROOT / "results/fe_exploracao"
    raw = fx.si.carregar_dados(fx.CFG["data_new"])
    pre = fx.temporal(raw)
    linhas = []
    for var in ["base", "vib", "vib_press", "temp_delta", "temp_rel"]:
        F = fx.features(pre, var)
        tr = F[(F.index >= fx.TS) & (F.index < fx.TE)]
        esc = fx.Escala().fit(tr.iloc[:-int(len(tr) * 0.15)])
        X = esc(F)
        for pt in sorted((out / "cands" / var).glob("*.pt")):
            m = fx.build_model("vae", X.shape[1], dense_layers=[128, 64, 32], latent_dim=16)
            m.load_state_dict(torch.load(pt, map_location="cpu")); m.eval()
            e = fx.erros(m, X).mean(axis=1)
            for modo in ("bruto", "suave"):
                s = e if modo == "bruto" else fx.suavizar(e)
                st = s[(s.index >= fx.TS) & (s.index < fx.TE)]
                thr = float(st.mean() + fx.Y_ALARM * st.std())
                f = fx.alertas(s, thr, raw)
                lin = {"variante": var, "cand": pt.stem, "modo": modo}
                for d in DATAS:
                    t0 = pd.Timestamp(d); tf = t0 + pd.Timedelta(hours=fx.RAMP); fim = tf + pd.Timedelta(hours=fx.HOLD)
                    janela = raw.loc[t0 - pd.Timedelta(days=5): fim + pd.Timedelta(hours=24)]
                    for k in (1, 2):
                        col = f"termica{k}x_{d[5:]}_h"
                        if f[(f.index >= t0) & (f.index <= fim)].any():
                            lin[col] = np.nan; continue
                        rj = fx.battery.inject(janela, TERMICA, t0, fx.RAMP, fx.HOLD, k)
                        ej = fx.erros(m, esc(fx.features(fx.temporal(rj), var))).mean(axis=1)
                        w = fx.alertas(ej if modo == "bruto" else fx.suavizar(ej), thr, rj).loc[t0:fim]
                        fi = w[w].index.min() if w.any() else None
                        lin[col] = round((tf - fi).total_seconds() / 3600, 1) if fi is not None else -99.0
                linhas.append(lin)
                print(lin, flush=True)
    pd.DataFrame(linhas).to_csv(out / "termica.csv", index=False)


if __name__ == "__main__":
    main()
