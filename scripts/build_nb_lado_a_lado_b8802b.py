"""Gera notebooks/b8802b/lado_a_lado_sem_temperatura_b8802b.ipynb: o modelo atual e o sem temperatura, lado a lado.

Lê o bundle de produção e um bundle do retreino sem temperatura (results/vp_vs_nivel, task 51984730); só roda inferência.

    python scripts/build_nb_lado_a_lado_b8802b.py && jupyter nbconvert --execute --inplace notebooks/b8802b/lado_a_lado_sem_temperatura_b8802b.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# B-8802B · modelo atual × modelo sem temperatura, lado a lado

Em todas as figuras: **à esquerda o modelo atual** (em produção, 5 sensores com a temperatura do mancal LA) e **à
direita o modelo sem temperatura** (vibrações LA/LNA + pressões de sucção e descarga), na mesma escala.

- **Linha de cima:** o sensor, com os **pontos vermelhos** onde o modelo está em alerta (a regra de produção: 15 de 20
  leituras acima do limiar e duração ≥ 1 h).
- **Linha de baixo:** o score do modelo dividido pelo limiar (acima de 1 = acima do limiar).

O modelo sem temperatura é um dos 12 do retreino completo (mesma arquitetura do atual: VAE 128-64-32, latente 16, taxa
1e-4), com a mesma janela de treino (2025) e a mesma regra de alarme.""")

code(r'''import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, matplotlib.dates as mdates
%config InlineBackend.figure_format = "retina"
RAIZ = Path.cwd().resolve()
while not (RAIZ / "scripts").exists(): RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ / "src"))
from transpetro_modelos.drift import battery
B = battery.BATTERY["B-8802B-2025"]; si = battery.si
raw = si.carregar_dados(B["data_new"])
raw22 = si.carregar_dados(next(Path(B["cross_era"]["csv"]).rglob(B["cross_era"]["csv_glob"])))
FALHA = pd.Timestamp(B["cross_era"]["failure"])
MODELOS = {"modelo atual (com temperatura)": (Path(B["bundle"]), "#2a78d6"),
           "modelo sem temperatura": (RAIZ / "results/vp_vs_nivel/bundles/vp/l128-64-32_lat16_lr0.0001_s1", "#1baf7a")}
VERMELHO, SERIE, TEXTO = "#e5484d", "#6f9fd1", "#52514e"
plt.rcParams.update({"font.size": 10.5})
def estilo(ax, t=None, size=11):
    if t: ax.set_title(t, loc="left", fontsize=size, fontweight="bold", color="#222222")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25, lw=0.5)

def rodar(bundle, df):
    torch.manual_seed(0)
    r = battery.score(bundle, df)
    thr = json.loads((bundle / "alarm.json").read_text())["threshold"]
    return r["reconstruction_error"] / thr, r["is_anomaly"].astype(bool)

def episodios(f):
    x = f[f]
    if x.empty: return []
    g = x.index.to_series().diff().gt(pd.Timedelta("1h")).cumsum()
    return [e.index[0] for _, e in x.groupby(g)]

def lado_a_lado(df, ini, fim, sensor, un, titulo, agreg=None, marcas=(), ylim_score=None):
    """Uma coluna por modelo: sensor com os alertas em cima, score ÷ limiar embaixo."""
    w = df.loc[pd.Timestamp(ini) - pd.Timedelta(days=2):fim]
    v = w[sensor].where(w["Pressão Descarga"] > 35).loc[ini:fim]
    fig, axes = plt.subplots(2, 2, figsize=(16, 6.2), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.5, 1]})
    resumo = {}
    for j, (nome, (b, cor)) in enumerate(MODELOS.items()):
        s, a = rodar(b, w); s, a = s.loc[ini:fim], a.loc[ini:fim]
        vv = v.resample(agreg).median() if agreg else v
        axes[0, j].plot(vv.index, vv.values, color=SERIE, lw=0.7)
        p = v.reindex(a.index)[a]; axes[0, j].scatter(p.index, p.values, color=VERMELHO, s=7, zorder=3)
        ss = s.resample(agreg).median() if agreg else s.groupby(s.index.to_series().diff().gt(pd.Timedelta("10min")).cumsum().values).transform(lambda x: x.rolling(12, min_periods=1).median())
        axes[1, j].plot(ss.index, ss.values, color=cor, lw=1.2); axes[1, j].axhline(1, color="black", ls="--", lw=1)
        ep = episodios(a)
        txt = f"{len(ep)} episódio(s) de alerta" + (": " + ", ".join(f"{t:%d/%m}" for t in ep[:6]) + ("…" if len(ep) > 6 else "") if ep else "")
        estilo(axes[0, j], f"{nome}\n{txt}")
        axes[1, j].set_ylabel("score ÷ limiar" if j == 0 else ""); axes[0, j].set_ylabel(un if j == 0 else "")
        for t, rot in marcas:
            for ax in axes[:, j]: ax.axvline(pd.Timestamp(t), color="#f0a020", ls="--", lw=1.3)
            axes[0, j].text(pd.Timestamp(t), 1.0, f" {rot}", transform=axes[0, j].get_xaxis_transform(), fontsize=9.5, color="#b5741a", va="top")
        resumo[nome] = (a, ep)
        estilo(axes[1, j])
    if ylim_score: axes[1, 0].set_ylim(*ylim_score)
    for ax in axes[1]: ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y" if (pd.Timestamp(fim) - pd.Timestamp(ini)).days > 60 else "%d/%m"))
    fig.suptitle(f"B-8802B · {titulo}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(); plt.show()
    return resumo''')

md("""## 1. 2025 a ago/2026 inteiro

Vibração do mancal LA (mediana de 6 h) e score (mediana de 6 h). É aqui que aparece a diferença de comportamento com o
mancal mais frio em abr–ago/2026.""")

code(r'''r_tudo = lado_a_lado(raw, "2025-01-01", "2026-08-11", "Vibração Bomba LA", "mm/s", "2025–2026: vibração do mancal LA, alertas e score",
                     agreg="6h", ylim_score=(0, 1.2))''')

code(r'''fig, axes = plt.subplots(1, 2, figsize=(16, 3.2), sharey=True)
t = raw["Temperatura Bomba LA"].where(raw["Pressão Descarga"] > 35).resample("6h").median()
for ax, (nome, (b, cor)) in zip(axes, MODELOS.items()):
    ax.plot(t.index, t.values, color="#eb6834", lw=0.9)
    entra = "entra no modelo" if "atual" in nome else "NÃO entra no modelo"
    estilo(ax, f"temperatura do mancal LA (°C) — {entra}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
axes[0].set_ylabel("°C"); fig.tight_layout()''')

code(r'''linhas = []
for nome, (b, cor) in MODELOS.items():
    s, a = rodar(b, raw)
    e25 = s.loc["2025"].mean()
    linhas.append({"modelo": nome, "score médio abr–ago/26 ÷ 2025": round(s.loc["2026-04":"2026-08"].mean() / e25, 2),
                   "% do tempo em alerta 2025": round(100 * a.loc["2025"].mean(), 3),
                   "% do tempo em alerta 2026": round(100 * a.loc["2026"].mean(), 3),
                   "episódios 2026": len(episodios(a.loc["2026"]))})
pd.DataFrame(linhas).set_index("modelo")''')

md("""**Leitura:** com o mancal esfriando (temperatura caindo de ~55 para ~45 °C a partir de abril/2026), o score do modelo
atual sobe (fica mais perto do limiar, sem alertar); o do modelo sem temperatura não muda. Os dois só alertam em janeiro/2026.""")

md("""## 2. Janeiro de 2026: os dois episódios de vibração""")

code(r'''_ = lado_a_lado(raw, "2026-01-07", "2026-01-21", "Vibração Bomba LNA", "mm/s", "7 a 21/jan/2026: vibração do mancal LNA, alertas e score")''')

md("""Nos dois episódios (10–12/01 e 17/01) a vibração fica acima do normal de 2025 — no de 10–12/01 a LNA vai ainda mais
alto que no de 17/01. O modelo sem temperatura alerta nos dois; o atual, só no de 17/01.""")

md("""## 3. Abril a agosto de 2026: o mancal mais frio""")

code(r'''_ = lado_a_lado(raw, "2026-03-15", "2026-08-11", "Temperatura Bomba LA", "°C", "mar–ago/2026: temperatura do mancal LA (pontos = alertas) e score",
                agreg="1h", ylim_score=(0, 1.2), marcas=[("2026-03-28", "mancal ~6 °C mais frio")])''')

md("""**Leitura:** depois do mancal esfriar, o score do modelo atual sobe e encosta no limiar várias vezes em jul–ago/2026 —
só não vira alerta por causa da persistência (15 de 20 leituras + 1 h). O do modelo sem temperatura fica baixo: para
ele, nada mudou, e de fato nada indica falha nesse período.""")

md("""## 4. A falha real de 2022""")

code(r'''_ = lado_a_lado(raw22, "2022-06-27", FALHA, "Vibração Bomba LA", "mm/s", "falha real de 06/07/2022: vibração do mancal LA, alertas e score",
                marcas=[(FALHA, "falha")])''')

md("""## 5. A falha simulada de 10/06/2026 (assinatura da falha de 2022 somada ao dado real)""")

code(r'''t0 = pd.Timestamp("2026-06-10"); tf = t0 + pd.Timedelta(hours=48)
inj = battery.inject(raw.loc[t0 - pd.Timedelta(days=6): tf + pd.Timedelta(hours=30)], B["synthetic"]["signature"], t0, 48, 24, 1.0)
_ = lado_a_lado(inj, str(t0 - pd.Timedelta(days=3)), str(tf + pd.Timedelta(hours=24)), "Vibração Bomba LA", "mm/s",
                "falha simulada a partir de 10/06/2026: vibração do mancal LA, alertas e score",
                marcas=[(t0, "início da rampa"), (tf, "fim da rampa")])''')

md("""**Leitura das falhas (seções 4 e 5):** os dois modelos se comportam igual — alertam ao mesmo tempo, porque a falha
está na vibração, que os dois têm.""")

md("""## 6. Os números lado a lado""")

code(r'''def antecedencia(bundle, df, t_fim, ini):
    s, a = rodar(bundle, df); a = a.loc[ini:t_fim]
    return round((t_fim - a[a].index.min()).total_seconds() / 3600) if a.any() else None
tab = {}
for nome, (b, cor) in MODELOS.items():
    s, a = rodar(b, raw)
    col = {"episódios de alerta em 2025": len(episodios(a.loc["2025"])), "episódios de alerta em 2026": len(episodios(a.loc["2026"])),
           "alerta em 10–12/01/26": "sim" if a.loc["2026-01-10":"2026-01-13"].any() else "não",
           "alerta em 17/01/26": "sim" if a.loc["2026-01-17 19:30":"2026-01-18 00:40"].any() else "não",
           "score abr–ago/26 ÷ 2025": round(s.loc["2026-04":"2026-08"].mean() / s.loc["2025"].mean(), 2),
           "falha real 2022 (h antes)": antecedencia(b, raw22, FALHA, "2022-06-29")}
    for d in ["2026-01-25", "2026-06-10", "2026-07-15"]:
        t0 = pd.Timestamp(d); tf = t0 + pd.Timedelta(hours=48)
        inj = battery.inject(raw.loc[t0 - pd.Timedelta(days=5): tf + pd.Timedelta(hours=24)], B["synthetic"]["signature"], t0, 48, 24, 1.0)
        v = antecedencia(b, inj, tf + pd.Timedelta(hours=24), t0)
        col[f"falha simulada {t0:%d/%m} (h antes do fim da rampa)"] = None if v is None else v - 24
    tab[nome] = col
pd.DataFrame(tab)''')

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/b8802b/lado_a_lado_sem_temperatura_b8802b.ipynb")
print("ok")
