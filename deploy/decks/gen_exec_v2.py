"""Deck EXECUTIVO (consulting clean) + figuras limpas. Gera HTML p/ preview e salva as
figuras como PNG p/ o pptx editável reusar. Storytelling linear, pouco texto, headlines
fortes, status por cor (verde/amarelo/vermelho), pipeline horizontal."""
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
SP = Path(os.environ.get("DECK_WORK", Path(__file__).resolve().parent / "_work"))
SP.mkdir(parents=True, exist_ok=True)
FIG = SP / "figs_exec_v2"; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "deploy")); sys.path.insert(0, str(ROOT / "src"))
from transpetro_modelos.data.loading import load_equipment_data  # noqa: E402
from simpred_inference import load_bundle, predict  # noqa: E402

INK = "#22303C"; GREEN = "#2E9E6E"; AMBER = "#E0A21E"; RED = "#CC3B3B"; GREY = "#9AA7B0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": "#CBD3D9", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#EDF1F4", "grid.linewidth": 1,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", format="png", dpi=130, bbox_inches="tight", transparent=True)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _debounced(above, n):
    return above.rolling(n, min_periods=n).sum().eq(n)


RES = {"B-8802B": ("2022-07-06", "2022-06-16"), "B-6511502A": ("2023-05-15", "2023-04-25")}


def result_fig(eq, name):
    falha, ne = RES[eq]
    d = ROOT / "deploy" / "Transpetro" / eq
    b = load_bundle(next((d / "modelos").glob("model_*")))
    r = predict(b, next((d / "dados").rglob("*_raw.csv")))
    thr = b.threshold; n = b.debounce_consecutive or 1
    fd = pd.Timestamp(falha)
    alarm = _debounced(r["reconstruction_error"] > thr, n)
    fp = 100 * _debounced(r[r.index < pd.Timestamp(ne)]["reconstruction_error"] > thr, n).mean()
    pre = r[(r.index >= fd - pd.Timedelta(days=10)) & (r.index < fd)]
    first = pre[_debounced(pre["reconstruction_error"] > thr, n)].index.min()
    lead = (fd - first) if first is not None else None
    win = r[(r.index >= fd - pd.Timedelta(days=10)) & (r.index <= fd + pd.Timedelta(days=1))]
    wal = win[alarm.reindex(win.index, fill_value=False)]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(win.index, win["reconstruction_error"], lw=1.1, color="#5B7384")
    ax.axhline(thr, color=RED, ls="--", lw=1.4)
    ax.scatter(wal.index, wal["reconstruction_error"], s=20, color=RED, zorder=3)
    ax.axvline(fd, color=INK, ls=":", lw=1.6)
    ax.text(fd, ax.get_ylim()[1] * 0.96, " falha", color=INK, fontsize=11, va="top")
    if first is not None:
        # label em área livre (canto sup. esquerdo) com caixa branca + seta ao 1º alarme
        ax.annotate("1º alarme", xy=(first, thr), xytext=(0.03, 0.9), textcoords="axes fraction",
                    fontsize=12, fontweight="bold", color=RED, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=RED, lw=1.3),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_ylabel("erro de reconstrução"); ax.margins(x=0.01)
    fig.autofmt_xdate(); fig.tight_layout()
    leadtxt = (str(lead).split(",")[0].replace("days", "dias").replace("day", "dia")
               if lead is not None else "—")
    return save(fig, name), dict(fp=fp, lead=leadtxt, first=str(first)[:10] if first is not None else None)


def donut_0302(name):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.pie([21, 7], colors=[RED, "#E4EAEE"], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.34, edgecolor="white", linewidth=3))
    ax.text(0, 0.12, "75%", ha="center", va="center", fontsize=46, fontweight="bold", color=RED)
    ax.text(0, -0.28, "sensores\nsem sinal útil", ha="center", va="center", fontsize=14, color=INK)
    ax.set(aspect="equal"); fig.tight_layout()
    return save(fig, name)


def bars_4064(name):
    anos = ["2024\n(antes)", "2025\n(pós-reparo)", "2026"]
    vals = [47, 68, 71]
    cols = [GREEN, AMBER, AMBER]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(anos, vals, color=cols, width=0.6)
    for bx, v in zip(bars, vals):
        ax.text(bx.get_x() + bx.get_width() / 2, v + 1.5, f"{v}°C", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(2, 71), xytext=(0, 47), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(1, 64, "+24°C", color=RED, fontsize=16, fontweight="bold", ha="center")
    ax.set_ylabel("Temp. mancal LNA (°C)"); ax.set_ylim(0, 85); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, name)


def bars_24001(name):
    estr = ["p99.9\n(refinado)", "p99.5\n(fixo)", "p90\n(sensível)"]
    vals = [36, 7892, 8742]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(estr, vals, color=[GREEN, AMBER, RED], width=0.6)
    for bx, v in zip(bars, vals):
        ax.text(bx.get_x() + bx.get_width() / 2, v + 150, f"{v:,}".replace(",", "."),
                ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("nº de alarmes"); ax.set_ylim(0, 9800); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, name)


def tradeoff_fig(name, dets, color):
    """Curva separabilidade: detecção pré-falha (%) vs falso positivo (%), do sweep do notebook."""
    fps = [5.0, 2.5, 1.0, 0.5, 0.3, 0.1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(fps, dets, "-o", color=color, lw=2.4, ms=8)
    i = fps.index(1.0)
    ax.scatter([1.0], [dets[i]], s=150, facecolor="white", edgecolor=color, lw=2.6, zorder=5)
    ax.annotate(f"{dets[i]:.0f}% @ 1% FP", (1.0, dets[i]), textcoords="offset points",
                xytext=(8, -28 if dets[i] > 55 else 12), fontsize=12.5, fontweight="bold", color=color)
    ax.set_xscale("log"); ax.set_xlim(0.08, 6.5); ax.set_ylim(0, 108)
    ax.set_xlabel("Falso positivo na janela normal (%)"); ax.set_ylabel("Detecção pré-falha (%)")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0,1", "0,2", "0,5", "1", "2", "5"])
    ax.grid(True); fig.tight_layout()
    return save(fig, name)


def draw_icon(kind, path):
    """Ícone monoline branco (fundo transparente) p/ assentar sobre o badge verde."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(1.4, 1.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    W = "#FFFFFF"; lw = 9

    def ln(xs, ys):
        ax.plot(xs, ys, color=W, lw=lw, solid_capstyle="round", solid_joinstyle="round")

    if kind == "data":  # base de dados (cilindro)
        ax.add_patch(mpatches.Ellipse((0.5, 0.76), 0.56, 0.18, fill=False, ec=W, lw=lw))
        ln([0.22, 0.22], [0.76, 0.28]); ln([0.78, 0.78], [0.76, 0.28])
        ax.add_patch(mpatches.Arc((0.5, 0.52), 0.56, 0.18, theta1=180, theta2=360, ec=W, lw=lw))
        ax.add_patch(mpatches.Arc((0.5, 0.28), 0.56, 0.18, theta1=180, theta2=360, ec=W, lw=lw))
    elif kind == "filter":  # funil
        ln([0.18, 0.82, 0.56, 0.56, 0.44, 0.44, 0.18], [0.80, 0.80, 0.50, 0.22, 0.22, 0.50, 0.80])
    elif kind == "select":  # sliders / seleção
        for y, kx in [(0.72, 0.62), (0.5, 0.36), (0.28, 0.68)]:
            ln([0.2, 0.8], [y, y]); ax.add_patch(mpatches.Circle((kx, y), 0.07, fc=W, ec=W))
    elif kind == "model":  # rede / autoencoder
        nodes = [(0.28, 0.72), (0.28, 0.28), (0.6, 0.5), (0.82, 0.72), (0.82, 0.28)]
        for a, b in [(0, 2), (1, 2), (2, 3), (2, 4)]:
            ln([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]])
        for x, y in nodes:
            ax.add_patch(mpatches.Circle((x, y), 0.075, fc=W, ec=W))
    elif kind == "detect":  # lupa
        ax.add_patch(mpatches.Circle((0.44, 0.58), 0.22, fill=False, ec=W, lw=lw))
        ln([0.60, 0.80], [0.42, 0.22])
    elif kind == "deploy":  # caixa / bundle
        ln([0.28, 0.72, 0.72, 0.28, 0.28], [0.24, 0.24, 0.68, 0.68, 0.24])
        ln([0.5, 0.5], [0.24, 0.68])
    fig.savefig(path, transparent=True, bbox_inches="tight", dpi=150)
    plt.close(fig)


print("gerando figuras...")
res8802_b, r8802 = result_fig("B-8802B", "res_8802")
res6511_b, r6511 = result_fig("B-6511502A", "res_6511")
d0302_b = donut_0302("donut_0302")
b4064_b = bars_4064("bars_4064")
b24001_b = bars_24001("bars_24001")
lara = {p: base64.b64encode((SP / f"lara_chart_{p}.png").read_bytes()).decode() for p in (2, 3, 4, 5)}
trade3403_b = tradeoff_fig("trade_3403c", [100, 100, 100, 100, 99.7, 98.8], GREEN)
trade90001_b = tradeoff_fig("trade_90001a", [99.7, 90.3, 48.0, 32.4, 26.2, 14.9], AMBER)
ICON_KEYS = ["data", "filter", "select", "model", "detect", "deploy"]
icons_b64 = {}
for _k in ICON_KEYS:
    _p = FIG / f"icon_{_k}.png"; draw_icon(_k, _p)
    icons_b64[_k] = base64.b64encode(_p.read_bytes()).decode()
print("res8802", r8802, "| res6511", r6511)

# ───────────────────────── HTML executivo ─────────────────────────
PIPE = [
    ("data", "Dados Brutos", ["CSV por sensor", "amostra de 30s a 1h"]),
    ("filter", "Pré-processamento", ["Remove ruído e estado parado", "resample · clip"]),
    ("select", "Seleção de variáveis", ["Sensores relevantes por equipamento", "+ resíduo Temp~Corrente (B-4064A)"]),
    ("model", "Modelagem", ["Autoencoder (VAE) via AutoML", "treina só em operação normal"]),
    ("detect", "Detecção", ["Erro &gt; limiar calibrado por FP", "validada em falhas históricas"]),
    ("deploy", "Deploy", ["Bundle pronto p/ integração", "alarme online a validar"]),
]
pipe_html = ""
for i, (key, t, bs) in enumerate(PIPE):
    bl = "".join(f"<li>{b}</li>" for b in bs)
    pipe_html += (f'<div class="pcard"><div class="picbadge"><img src="data:image/png;base64,{icons_b64[key]}"></div>'
                  f'<div class="pt">{t}</div><ul>{bl}</ul></div>')
    if i < len(PIPE) - 1:
        pipe_html += '<div class="parrow">&#8594;</div>'

CSS = """
* { box-sizing:border-box; }
:root { --ink:#22303C; --green:#2E9E6E; --amber:#E0A21E; --red:#CC3B3B; --line:#E2E6EA; --soft:#F5F7F9; }
body { margin:0; font-family:"Segoe UI",Arial,Helvetica,sans-serif; color:var(--ink); background:#d9dde1; padding:18px 0; }
.slide { width:min(1280px,96vw); aspect-ratio:16/9; margin:20px auto; background:#fff; position:relative;
  box-shadow:0 6px 22px rgba(0,0,0,.14); display:flex; flex-direction:column; padding:3.4% 4%; overflow:hidden; }
.kicker { color:var(--green); font-size:clamp(10px,1.05vw,14px); font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin:0 0 1.2%; }
h1.t { font-size:clamp(20px,2.7vw,36px); font-weight:800; margin:0 0 1.4%; line-height:1.12; }
.rule { height:3px; width:54px; background:var(--green); margin-bottom:2.6%; }
.foot { position:absolute; left:4%; right:4%; bottom:2.4%; display:flex; justify-content:space-between;
  color:#9AA7B0; font-size:clamp(8px,.95vw,12px); border-top:1px solid var(--line); padding-top:1%; }
.body { flex:1; display:flex; gap:3%; min-height:0; }
/* chips de status */
.chip { display:inline-block; padding:3px 12px; border-radius:20px; font-size:clamp(10px,1.1vw,14px); font-weight:700; color:#fff; }
.c-green{background:var(--green);} .c-amber{background:var(--amber);} .c-red{background:var(--red);}
/* callout */
.callout { background:var(--soft); border-left:5px solid var(--green); padding:2.2% 2.6%; border-radius:0 8px 8px 0;
  font-size:clamp(12px,1.4vw,19px); font-weight:700; }
.callout.amber{border-color:var(--amber);} .callout.red{border-color:var(--red);}
ul.lean { margin:2% 0 0; padding-left:1.1em; } ul.lean li{ font-size:clamp(12px,1.45vw,19px); line-height:1.5; margin-bottom:2.2%; }
.imgcol{ flex:1.25; display:flex; align-items:center; justify-content:center; } .imgcol img{ max-width:100%; max-height:100%; object-fit:contain; }
.txtcol{ flex:1; display:flex; flex-direction:column; justify-content:center; }
/* capa */
.cover{ justify-content:center; }
.cover .big{ font-size:clamp(40px,6.6vw,92px); font-weight:800; letter-spacing:-2px; margin:0; }
.cover .sub{ font-size:clamp(15px,1.9vw,24px); color:#5B7384; margin:1.4% 0 0; }
.cover .brandlock{ position:absolute; top:7%; right:5.5%; display:flex; align-items:center; gap:12px; }
.cover .brandmk{ width:56px; height:56px; border-radius:14px; background:var(--green); color:#fff; font-size:26px;
  display:flex; align-items:center; justify-content:center; }
.cover .brandtx{ display:flex; flex-direction:column; gap:4px; line-height:1; }
.cover .brandtx b{ font-weight:800; font-size:23px; color:var(--ink); }
.cover .brandtx span{ font-weight:600; font-size:12px; color:#9AA7B0; letter-spacing:.04em; }
/* pipeline */
.piperow{ display:flex; align-items:stretch; gap:.4%; }
.pcard{ flex:1; background:#fff; border:1px solid var(--line); border-radius:12px; padding:1.4% 1.1%; display:flex; flex-direction:column; }
.picbadge{ width:clamp(36px,4.3vw,56px); aspect-ratio:1; background:var(--green); border-radius:12px;
  display:flex; align-items:center; justify-content:center; }
.picbadge img{ width:60%; height:60%; object-fit:contain; }
.pcard .pt{ font-weight:800; font-size:clamp(11px,1.25vw,16px); margin:6% 0 5%; }
.pcard ul{ margin:0; padding-left:1em; } .pcard li{ font-size:clamp(9px,1.02vw,13px); color:#5B7384; line-height:1.3; margin-bottom:5%; }
.parrow{ display:flex; align-items:center; color:var(--green); font-weight:800; font-size:clamp(12px,1.5vw,20px); }
.sidebox{ margin-top:2.4%; display:flex; gap:3%; }
.sidebox table{ border-collapse:collapse; width:62%; } .sidebox th,.sidebox td{ border:1px solid var(--line); padding:.7% 1.1%; font-size:clamp(9px,1.05vw,13px); text-align:left; }
.sidebox th{ background:var(--soft); font-weight:700; }
.sidenote{ width:38%; color:#9AA7B0; font-size:clamp(9px,1vw,13px); font-style:italic; align-self:center; }
/* dashboard */
.dash{ flex:1; display:flex; gap:2.4%; }
.dcol{ flex:1; border:1px solid var(--line); border-radius:12px; padding:2%; display:flex; flex-direction:column; }
.dcol .dh{ font-weight:800; font-size:clamp(12px,1.45vw,19px); padding:3% 0; text-align:center; border-radius:8px; color:#fff; margin-bottom:6%; }
.dcol.green .dh{background:var(--green);} .dcol.amber .dh{background:var(--amber);} .dcol.red .dh{background:var(--red);}
.dcol .item{ border-bottom:1px solid var(--line); padding:5% 2%; } .dcol .item:last-child{border-bottom:none;}
.dcol .eq{ font-weight:800; font-size:clamp(12px,1.4vw,18px); } .dcol .ds{ color:#5B7384; font-size:clamp(10px,1.15vw,15px); margin-top:3%; }
/* divisor de apêndice */
.appleft{ flex:0 0 42%; background:var(--green); color:#fff; padding:4.5% 3.6%; display:flex; flex-direction:column; justify-content:center; }
.appkick{ font-size:clamp(11px,1.1vw,15px); font-weight:700; letter-spacing:.18em; opacity:.85; margin:0 0 6%; }
.apptitle{ font-size:clamp(34px,5vw,68px); font-weight:800; margin:0; letter-spacing:-1px; }
.appsub{ font-size:clamp(14px,1.7vw,22px); font-weight:700; margin:3% 0 0; opacity:.95; }
.appmk{ margin:auto 0 0; font-weight:800; font-size:clamp(13px,1.4vw,18px); opacity:.92; }
.appright{ flex:1; padding:4.5% 4%; display:flex; flex-direction:column; justify-content:center; }
.apphead{ font-size:clamp(15px,1.7vw,22px); font-weight:800; color:var(--ink); margin:0 0 5%; }
.appitem{ display:flex; align-items:center; gap:14px; margin-bottom:3.4%; }
.appnum{ flex:0 0 auto; width:34px; height:34px; border-radius:50%; background:var(--green); color:#fff;
  font-weight:800; font-size:16px; display:flex; align-items:center; justify-content:center; }
.appt{ font-weight:800; font-size:clamp(13px,1.5vw,19px); color:var(--ink); }
.appd{ color:#5B7384; font-size:clamp(11px,1.25vw,16px); margin-top:2px; }
"""


def foot(n):
    return f'<div class="foot"><span>SimPred · Manutenção Preditiva · Transpetro</span><span>{n}</span></div>'


def headed(kicker, title, body_html, n):
    return (f'<section class="slide"><p class="kicker">{kicker}</p><h1 class="t">{title}</h1>'
            f'<div class="rule"></div><div class="body">{body_html}</div>{foot(n)}</section>')


def imgslide(kicker, title, img_b64, chip, lead_callout, bullets, n, callout_cls=""):
    bl = "".join(f"<li>{b}</li>" for b in bullets)
    body = (f'<div class="imgcol"><img src="data:image/png;base64,{img_b64}"></div>'
            f'<div class="txtcol"><div><span class="chip {chip}">{lead_callout[0]}</span></div>'
            f'<div class="callout {callout_cls}" style="margin-top:5%">{lead_callout[1]}</div>'
            f'<ul class="lean">{bl}</ul></div>')
    return headed(kicker, title, body, n)


slides = []
# 1 capa
slides.append(
    f'<section class="slide cover"><p class="kicker">SimPred · Transpetro · junho 2026</p>'
    f'<p class="big">Manutenção Preditiva</p>'
    f'<p class="sub">Detecção de falhas em bombas a partir dos dados de operação<br>'
    f'Francisco Colatino de Lima · Lara Fernanda Amorim A. Cavalcante</p>'
    f'<div class="brandlock"><div class="brandmk">&#9670;</div>'
    f'<div class="brandtx"><b>SimPred</b><span>MANUTENÇÃO PREDITIVA</span></div></div>'
    f'{foot("01")}</section>')

# 2 pipeline
pipe_body = (
    f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center">'
    f'<div class="piperow">{pipe_html}</div>'
    f'<div class="sidebox"><table>'
    f'<tr><th>Equip.</th><th>resample</th><th>filtro de estado</th><th>passo extra</th></tr>'
    f'<tr><td>B-8802B</td><td>5 min</td><td>P. Descarga &gt; 35</td><td>—</td></tr>'
    f'<tr><td>B-6511502A</td><td>5 min</td><td>Corrente &gt; 60</td><td>—</td></tr>'
    f'<tr><td>B-4064A</td><td>1 h</td><td>Corrente &gt; 5</td><td>resíduo Temp~Corrente</td></tr>'
    f'</table><div class="sidenote">*Mesmo esqueleto para todos; só os parâmetros mudam por equipamento. '
    f'O B-4064A troca a temperatura pelo resíduo Temp~Corrente (regime térmico).</div></div></div>')
slides.append(headed("Como funciona", "Do dado bruto à detecção, em 6 etapas", pipe_body, "02"))

# 3 resultados (2 charts)
res_body = (
    f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center">'
    f'<div class="callout">Detectamos as 2 falhas conhecidas com dias de antecedência — e quase zero alarme falso.</div>'
    f'<div style="display:flex;gap:3%;margin-top:2.5%">'
    f'<div style="flex:1;text-align:center"><img src="data:image/png;base64,{res8802_b}" style="max-width:100%;max-height:38vh"><br>'
    f'<b>B-8802B</b> · alarme {r8802["first"]} · FP {r8802["fp"]:.2f}%</div>'
    f'<div style="flex:1;text-align:center"><img src="data:image/png;base64,{res6511_b}" style="max-width:100%;max-height:38vh"><br>'
    f'<b>B-6511502A</b> · alarme {r6511["first"]} · FP {r6511["fp"]:.2f}%</div>'
    f'</div></div>')
slides.append(headed("Resultados", "2 de 8 equipamentos prontos para produção", res_body, "03"))

# 4-5 NOVOS: B-3403C e B-90001A (análise Lara, dados interpolados)
slides.append(imgslide(
    "Detecção validada · análise Lara", "B-3403C: sinal forte ~3 semanas antes da restrição",
    trade3403_b, "c-green",
    ("B-3403C · Detecção validada (a empacotar)",
     "Baseline quieto por meses e rampa clara ~3 semanas antes do evento (12/Set/2023)."),
    ["Detecção pré-falha ~100% mesmo apertando o limiar (p99 → 100% @ 1% de FP).",
     "AutoML (autoencoder DENSE, 611 trials) sobre dados interpolados.",
     "Validado em 1 evento · FP ainda in-sample → validar fora da amostra e empacotar."],
    "04"))
slides.append(imgslide(
    "Requer validação · análise Lara", "B-90001A: detecção defensável, mas margem estreita",
    trade90001_b, "c-amber",
    ("B-90001A · Requer validação",
     "Detecta ~1 mês antes da falha (28/Ago/2021), mas a separação é estreita."),
    ["No mesmo 1% de FP, detecta só 48% — o B-3403C detecta 100%.",
     "Com debounce agressivo chega a ~85% @ 1,6% de FP (melhor equilíbrio).",
     "AutoML Isolation Forest · 1 evento · validar fora da amostra."],
    "05", "amber"))

# 4 qualidade de dados — tabela status
dq_body = (
    f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center">'
    f'<div class="callout red">A qualidade do dado de operação — não o modelo — é o que limita os demais equipamentos.</div>'
    f'<table style="border-collapse:collapse;width:100%;margin-top:2.5%">'
    f'<tr style="background:var(--soft)"><th style="border:1px solid var(--line);padding:1%;text-align:left">Equipamento</th>'
    f'<th style="border:1px solid var(--line);padding:1%;text-align:left">Situação do dado</th>'
    f'<th style="border:1px solid var(--line);padding:1%;text-align:center">Status</th></tr>'
    f'<tr><td style="border:1px solid var(--line);padding:1%"><b>B-0302C</b></td><td style="border:1px solid var(--line);padding:1%">75% dos sensores sem sinal · 93% parada</td><td style="border:1px solid var(--line);padding:1%;text-align:center"><span class="chip c-red">Bloqueado</span></td></tr>'
    f'<tr><td style="border:1px solid var(--line);padding:1%"><b>B-4703.24001B</b></td><td style="border:1px solid var(--line);padding:1%">76% do tempo parada</td><td style="border:1px solid var(--line);padding:1%;text-align:center"><span class="chip c-red">Bloqueado</span></td></tr>'
    f'<tr><td style="border:1px solid var(--line);padding:1%"><b>B-4064A</b></td><td style="border:1px solid var(--line);padding:1%">Reconstruída em 2024/25 · mancal +24°C</td><td style="border:1px solid var(--line);padding:1%;text-align:center"><span class="chip c-amber">Validar</span></td></tr>'
    f'<tr><td style="border:1px solid var(--line);padding:1%"><b>B-24001B</b></td><td style="border:1px solid var(--line);padding:1%">Dados intermitentes · threshold instável</td><td style="border:1px solid var(--line);padding:1%;text-align:center"><span class="chip c-amber">Validar</span></td></tr>'
    f'</table></div>')
slides.append(headed("Qualidade de dados", "O que separa os bons dos fracos", dq_body, "06"))

# 5 B-0302C
slides.append(imgslide("Bloqueado por dados", "75% dos sensores não possuem sinal útil", d0302_b,
                       "c-red", ("B-0302C · Bloqueado",
                                 "Sem instrumentação válida, não há o que o modelo aprenda."),
                       ["Só 7 de 28 canais têm sinal real.",
                        "A bomba aparece parada em 93% das leituras.",
                        "Pergunta à operação: sensores descomissionados ou falha de coleta?"], "07", "red"))

# 6 B-4064A
slides.append(imgslide("Requer validação", "A reconstrução da bomba mudou o regime térmico", b4064_b,
                       "c-amber", ("B-4064A · Validar",
                                   "O modelo de 2024 acusa falso positivo no equipamento reconstruído."),
                       ["Falha 08/2024 → reconstruída na Sulzer (peças de outra bomba).",
                        "Mancal passa a operar +24°C acima do regime anterior.",
                        "Ação: re-baselinar a partir de 13/01/2025."], "08", "amber"))

# 7 B-24001B (minha versão)
slides.append(imgslide("Requer validação", "Sem dado estável, o alarme varia de 36 a 8.700", b24001_b,
                       "c-amber", ("B-24001B · Validar",
                                   "Dados intermitentes tornam o threshold instável (análise Lara)."),
                       ["Falha 06/01/2025 — vibração no mancal LNA.",
                        "O nº de alarmes muda ~240× só trocando o limiar.",
                        "Ação: validar a densidade/continuidade da coleta."], "09", "amber"))

# 8 dashboard executivo
dash_body = (
    '<div class="dash">'
    '<div class="dcol green"><div class="dh">Prontos para produção</div>'
    '<div class="item"><div class="eq">B-8802B</div><div class="ds">Alarme dias antes · FP 0,06%</div></div>'
    '<div class="item"><div class="eq">B-6511502A</div><div class="ds">Alarme dias antes · FP 0,05%</div></div></div>'
    '<div class="dcol amber"><div class="dh">Requer validação</div>'
    '<div class="item"><div class="eq">B-3403C</div><div class="ds">Sinal forte · 1 evento · a empacotar</div></div>'
    '<div class="item"><div class="eq">B-90001A</div><div class="ds">Detecção defensável · margem estreita</div></div>'
    '<div class="item"><div class="eq">B-4064A</div><div class="ds">Re-baselinar pós-reconstrução</div></div>'
    '<div class="item"><div class="eq">B-24001B</div><div class="ds">Threshold instável · validar coleta</div></div></div>'
    '<div class="dcol red"><div class="dh">Bloqueados por dados</div>'
    '<div class="item"><div class="eq">B-0302C</div><div class="ds">75% dos sensores sem sinal</div></div>'
    '<div class="item"><div class="eq">B-4703.24001B</div><div class="ds">76% do tempo parada</div></div></div>'
    '</div>')
slides.append(headed("Onde estamos", "Status executivo dos 8 equipamentos", dash_body, "10"))

# 9 divisória apêndice (divisor de seção: painel verde + agenda das estratégias)
APP_ITEMS = [
    ("1", "Otsu por variável", "corte automático por sensor"),
    ("2", "p90 dinâmico", "8.742 alarmes — muito sensível"),
    ("3", "p99.9", "36 alarmes — conservador"),
    ("4", "p99.5 fixo", "7.892 alarmes — poucas detecções"),
]
app_list = "".join(
    f'<div class="appitem"><div class="appnum">{n}</div>'
    f'<div><div class="appt">{t}</div><div class="appd">{d}</div></div></div>' for n, t, d in APP_ITEMS)
slides.append(
    f'<section class="slide" style="padding:0;flex-direction:row">'
    f'<div class="appleft"><p class="appkick">APÊNDICE</p><p class="apptitle">B-24001B</p>'
    f'<p class="appsub">Análise original — Lara</p><p class="appmk">&#9670; SimPred</p></div>'
    f'<div class="appright"><p class="apphead">Estratégias de threshold testadas (AutoML)</p>'
    f'{app_list}<div class="callout amber" style="margin-top:4%">O nº de alarmes varia ~240× '
    f'conforme o limiar — por isso a recomendação de validar os dados.</div></div></section>')

# 10-13 estratégias de threshold (B-24001B, Lara) no nosso template
LARA_SLIDES = [
    (2, "Estratégia 1 — Otsu por variável",
     "Threshold automático (Otsu) em cada uma das 6 variáveis de vibração.",
     ["Define um corte por sensor, sem ajuste manual.", "Ponto de partida do estudo."]),
    (3, "Estratégia 2 — p90 dinâmico: 8.742 alarmes",
     "Muito sensível: detecta antes do desligamento, mas dispara demais.",
     ["Debounce 1/1 (sem suavização).", "Excesso de alarmes ao longo da série."]),
    (4, "Estratégia 3 — p99.9: 36 alarmes",
     "Conservador: detecção reduzida, comportamento parecido.",
     ["Refinamento dos parâmetros.", "Poucos alarmes — risco de perder sinal."]),
    (5, "Estratégia 4 — p99.5 fixo: 7.892 alarmes",
     "Threshold fixo (debounce 10/12): poucas detecções próximas à falha.",
     ["Resultado semelhante ao dinâmico.", "Confirma a instabilidade conforme o limiar."]),
]
for k, (p, title, call, bs) in enumerate(LARA_SLIDES):
    slides.append(imgslide("Apêndice · B-24001B (Lara)", title, lara[p], "c-amber",
                           ("B-24001B · Validar", call), bs, f"{11 + k:02d}", "amber"))

HTML = (f'<!DOCTYPE html><html lang="pt-br"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>SimPred — Manutenção Preditiva Transpetro</title><style>{CSS}</style></head>'
        f'<body>{"".join(slides)}</body></html>')
out = ROOT / "deploy" / "slides_executivo_transpetro_v2.html"
out.write_text(HTML, encoding="utf-8")
print("escrito:", out, f"({len(HTML)//1024} KB) · {len(slides)} slides")
