"""
Catálogo de comportamentos do B-8802B (out/2026): o modelo atual (5 sensores) e o sem temperatura (4), lado a lado.

Em cada situação, o que se espera do modelo e o que ele faz (% do tempo em alerta, 1º alerta). As mudanças são
aplicadas ao dado real de 2025 (o período do treino, de operação normal conhecida) a partir de 01/09/2025 por 60 dias,
só com a bomba operando; sensor zerado e travado, em 01/10/2025. Mais: os episódios de jan/2026, o mancal real mais frio
de abr–ago/2026, a falha real de 2022 e a falha simulada (assinatura de 2022) em 5 datas de 2026.

    python scripts/catalogo_b8802b.py        # ~5 min local, só inferência
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transpetro_modelos.drift import battery  # noqa: E402

B = battery.BATTERY["B-8802B-2025"]
MODELOS = {"atual (5 sensores)": Path(B["bundle"]),
           "sem temperatura (4)": ROOT / "deploy_v2/Transpetro/B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE_sem_temp"}
OUT = ROOT / "results/b8802b_v2"
D0, D1 = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-31")
VLA, VLNA, TLA = "Vibração Bomba LA", "Vibração Bomba LNA", "Temperatura Bomba LA"
DATAS_SINT = ["2026-01-25", "2026-03-05", "2026-06-10", "2026-07-15", "2026-08-01"]
CATALOGO = {   # nome → (mudança, início, fim da mudança, fim da janela, esperado)
    "normal (2025, sem mudança)": ({}, D0, D1, D1, "não alerta"),
    "mancal LA −8 °C (mais frio)": ({TLA: ("+", -8.0)}, D0, D1, D1, "não alerta (temperatura: monitor M8)"),
    "mancal LA +8 °C (mais quente)": ({TLA: ("+", 8.0)}, D0, D1, D1, "não alerta (temperatura: monitor M8 avisa)"),
    "vibração LA +10%": ({VLA: ("*", 1.1)}, D0, D1, D1, "pode alertar (sensibilidade)"),
    "vibração LA +20%": ({VLA: ("*", 1.2)}, D0, D1, D1, "pode alertar (sensibilidade)"),
    "vibração LA +30%": ({VLA: ("*", 1.3)}, D0, D1, D1, "alerta"),
    "vibração LA +50%": ({VLA: ("*", 1.5)}, D0, D1, D1, "alerta"),
    "vibrações LA e LNA ×1,3": ({VLA: ("*", 1.3), VLNA: ("*", 1.3)}, D0, D1, D1, "alerta"),
    "outro regime (descarga −5 bar)": ({"Pressão Descarga": ("+", -5.0)}, D0, D1, D1, "não alerta (normal novo: retreino)"),
    "sensor de vibração LA zerado (1 dia)": ({VLA: ("=", 0.0)}, pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-02"),
                                            pd.Timestamp("2025-10-03"), "alerta (falha de instrumento)"),
    "sensor de vibração LA travado (2 dias)": ({VLA: ("trava", None)}, pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-03"),
                                              pd.Timestamp("2025-10-04"), "não alerta (dado congelado: monitor M7)"),
}


def mudar(raw, ini, fim, mud):
    out = raw.copy()
    m = (out.index >= ini) & (out.index < fim) & (out["Pressão Descarga"] > 35)
    for c, (op, v) in mud.items():
        if op == "+": out.loc[m, c] += v
        elif op == "*": out.loc[m, c] *= v
        elif op == "=": out.loc[m, c] = v
        else: out.loc[m, c] = out.loc[out.index < ini, c].dropna().iloc[-1]
    return out


def alerta(b, df):
    torch.manual_seed(0)
    return battery.score(b, df)["is_anomaly"].astype(bool)


def score(b, df):
    import json
    torch.manual_seed(0)
    return battery.score(b, df)["reconstruction_error"] / json.loads((b / "alarm.json").read_text())["threshold"]


def episodios(f):
    x = f[f]
    return 0 if x.empty else int((x.index.to_series().diff() > pd.Timedelta("1h")).sum() + 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = battery.si.carregar_dados(B["data_new"])
    janela = raw.loc["2025-08-01":"2025-11-15"]
    ce = B["cross_era"]; raw22 = battery.si.carregar_dados(next(Path(ce["csv"]).rglob(ce["csv_glob"])))
    falha, r0 = pd.Timestamp(ce["failure"]), pd.Timestamp(ce["ramp_start"])
    sy = B["synthetic"]
    linhas = []
    for nome_m, b in MODELOS.items():
        for nome, (mud, ini, fim_m, fim_j, esp) in CATALOGO.items():
            f = alerta(b, mudar(janela, ini, fim_m, mud)).loc[ini:fim_j]
            t1 = f[f].index.min() if f.any() else None
            linhas.append({"modelo": nome_m, "situação": nome, "esperado": esp, "% do tempo em alerta": round(100 * f.mean(), 2),
                           "episódios": episodios(f), "1º alerta (h depois)": round((t1 - ini).total_seconds() / 3600, 1) if t1 is not None else None})
        f = alerta(b, raw)
        for nome, (a, z), esp in [("episódio 10–12/01/26 (vibração)", ("2026-01-10 18:00", "2026-01-12 12:00"), "alerta (evento esperado)"),
                                  ("episódio 17/01/26 (vibração)", ("2026-01-17 21:30", "2026-01-17 22:40"), "alerta (evento esperado)")]:
            w = f.loc[pd.Timestamp(a) - pd.Timedelta(hours=2): pd.Timestamp(z) + pd.Timedelta(hours=2)]
            linhas.append({"modelo": nome_m, "situação": nome, "esperado": esp, "% do tempo em alerta": round(100 * w.mean(), 2),
                           "episódios": episodios(w), "1º alerta (h depois)": 0.0 if w.any() else None})
        s = score(b, raw)
        w = f.loc["2026-04-01":"2026-08-11"]
        linhas.append({"modelo": nome_m, "situação": "mancal real ~6 °C mais frio (abr–ago/26)", "esperado": "não alerta",
                       "% do tempo em alerta": round(100 * w.mean(), 2), "episódios": episodios(w), "1º alerta (h depois)": None,
                       "score ÷ limiar (mediana · p99)": f"{s.loc['2026-04':'2026-08'].median():.2f} · {s.loc['2026-04':'2026-08'].quantile(.99):.2f}"})
        f22 = alerta(b, raw22); p = f22.loc[r0:falha]; t1 = p[p].index.min() if p.any() else None
        linhas.append({"modelo": nome_m, "situação": "falha real de 06/07/2022", "esperado": "alerta antes da falha",
                       "antecedência (h)": round((falha - t1).total_seconds() / 3600, 1) if t1 is not None else None})
        for d in DATAS_SINT:
            t0 = pd.Timestamp(d); tf = t0 + pd.Timedelta(hours=sy["ramp_h"]); fim = tf + pd.Timedelta(hours=sy["hold_h"])
            base = f.loc[t0:fim].any()
            j = raw.loc[t0 - pd.Timedelta(days=4): fim + pd.Timedelta(hours=12)]
            g = alerta(b, battery.inject(j, sy["signature"], t0, sy["ramp_h"], sy["hold_h"], 1.0)).loc[t0:fim]
            t1 = g[g].index.min() if g.any() else None
            linhas.append({"modelo": nome_m, "situação": f"falha simulada em {t0:%d/%m/%Y}", "esperado": "alerta antes do fim da rampa (48 h)",
                           "antecedência (h)": round((tf - t1).total_seconds() / 3600, 1) if t1 is not None else None,
                           "base_ja_alertava": bool(base),
                           "bomba_operando_%": round(100 * (raw.loc[t0:fim, "Pressão Descarga"] > 35).mean(), 1)})
        print(nome_m, "ok", flush=True)
    t = pd.DataFrame(linhas); t.to_csv(OUT / "catalogo.csv", index=False)
    pd.set_option("display.width", 250); print(t.to_string())


if __name__ == "__main__":
    main()
