"""Plot saved B8802B experiments as aligned sensor/score/alert time series.

Run: .venv/bin/python scripts/b8802b_timeline_report.py
No fitting or inference: score and flag columns are read verbatim from parquet.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
import numpy as np
import pandas as pd

from b8802b_feature_experiment import ROOT, BUNDLE, FAILURE, load_data, prepare, si

OUT = ROOT / "reports/b8802b_candidate_results"
LEVELS = ROOT / "results/b8802b_feature_candidates_levels"
INITIAL = ROOT / "results/b8802b_feature_candidates"
NOTEBOOK = ROOT / "notebooks/b8802b/comparacao_candidatos_series_temporais.ipynb"
MODELS = {
    "VAE_atual": "VAE atual (bundle 2025)",
    "C1_vibracao": "C1 · vibração",
    "C2_pressao": "C2 · + pressões",
    "C3_dinamica_termica": "C3 · + dinâmica térmica",
    "C1_conservative_exploratory": "C1 conservador · exploratório",
}
SENSORS = [
    ("Vibração Bomba LA", "Vibração LA\n(mm/s)"),
    ("Vibração Bomba LNA", "Vibração LNA\n(mm/s)"),
    ("Temperatura Bomba LA", "Temperatura LA\n(°C)"),
    ("Pressão Sucção", "Pressão\nsucção"),
    ("Pressão Descarga", "Pressão\ndescarga"),
]
VIEWS = [
    ("visao_2026", "2026: visão completa da avaliação", "recent", "2026-01-01", "2026-08-11", "levels",
     "Visão geral: observe a carga de alertas ao longo dos meses. Os intervalos sem operação válida aparecem como lacunas."),
    ("janeiro", "Janeiro de 2026: onde os modelos divergem", "recent", "2026-01-01", "2026-02-01", "levels",
     "Os candidatos da ablação sinalizam mais instantes que o VAE. Isso não permite classificá-los como falhas ou falsos positivos sem a operação."),
    ("janeiro_11", "11/jan: episódio adicional do C1 conservador", "recent", "2026-01-10", "2026-01-12", "levels",
     "O C1 conservador registra 5,25 horas com flag de alerta, entre 00:15 e 06:00 de 11/jan, com interrupções. O VAE não gera alerta operacional nessa janela. O evento precisa de classificação operacional."),
    ("janeiro_17", "17/jan: episódio também sinalizado pelo VAE", "recent", "2026-01-17", "2026-01-19", "levels",
     "Compare o crescimento da vibração com a entrada e a duração de cada sinalização. Pontos acima do limiar e alerta operacional são etapas diferentes."),
    ("historico_2022", "2022: histórico anterior e falha conhecida", "old", "2022-05-15", "2022-07-10", "levels",
     "A linha vertical marca a falha conhecida de 06/jul às 10:00. Alertas anteriores fora da aproximação da falha precisam ser examinados; o primeiro alerta de C3 não comprova maior antecipação útil."),
    ("falha_2022", "2022: aproximação da falha conhecida", "old", "2022-06-29", "2022-07-07", "levels",
     "C1 conservador preserva aproximadamente 49,75 h de antecedência. Este replay histórico já foi explorado e não constitui um teste independente."),
    ("inicial_janeiro", "Primeira montagem: nível + tendência, janeiro", "recent", "2026-01-01", "2026-02-01", "initial",
     "Esta é a primeira rodada, antes da ablação. Seus candidatos perderam o evento de janeiro; os limiares mais altos ajudam a visualizar o motivo."),
    ("inicial_falha", "Primeira montagem: nível + tendência, falha", "old", "2022-06-29", "2022-07-07", "initial",
     "A primeira montagem ainda sinaliza a falha forte de 2022. Preservar esse evento isolado não foi suficiente para aprovar os candidatos."),
]


def load_artifacts():
    recent, old = load_data()
    data = {}
    for period, raw in [("recent", recent), ("old", old)]:
        sensors = prepare(raw)
        real = si._instantes_com_operacao(BUNDLE, raw, sensors.index)
        rounds = {}
        for design, folder in [("levels", LEVELS), ("initial", INITIAL)]:
            thresholds = json.loads((folder / "thresholds.json").read_text())
            names = list(MODELS) if design == "levels" else list(MODELS)[:4]
            if design == "levels":
                trade = pd.read_csv(folder / "threshold_tradeoff_exploratory.csv")
                thresholds[names[-1]] = float(trade.loc[trade.model.eq("C1_vibracao") & trade.q.eq(1), "threshold"].item())
            scores = {}
            for name in names:
                suffix = "_2022_scores.parquet" if period == "old" else "_scores.parquet"
                frame = pd.read_parquet(folder / (name + suffix))
                assert frame.index.equals(sensors.index), (period, name, "index mismatch")
                assert frame.index.is_unique and frame.index.is_monotonic_increasing
                assert frame[["alarm", "alert"]].dtypes.eq(bool).all()
                assert not (frame.alert & ~frame.alarm).any()
                scores[name] = frame
            rounds[design] = {"scores": scores, "thresholds": thresholds}
        data[period] = {"sensors": sensors, "real": real, "rounds": rounds}
    return data


def cut(frame, start, end):
    """Use half-open intervals, including in tables and plot markers."""
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def with_gaps(series):
    """Insert NaNs so matplotlib never connects across missing 5-minute samples."""
    if series.empty:
        return series
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="5min"))


def window_summary(data, period, start, end, design="levels"):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rounds = data[period]["rounds"][design]
    rows = []
    for name, frame in rounds["scores"].items():
        f = cut(frame, start, end)
        real = data[period]["real"].reindex(f.index)
        flagged = f.index[f.alert]
        rows.append({"Modelo": MODELS[name], "Limiar": rounds["thresholds"][name],
                     "Acima do limiar (pontos válidos)": int(((f.score > rounds["thresholds"][name]) & real).sum()),
                     "Alarme (pontos)": int(f.alarm.sum()), "Alerta (pontos)": int(f.alert.sum()),
                     "Horas com flag de alerta": float(f.alert.sum()/12),
                     "Primeiro alerta na janela": str(flagged.min()) if len(flagged) else "—"})
    return pd.DataFrame(rows)


def plot_window(data, period, start, end, design="levels", title="B8802B"):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    block = data[period]
    sensors = cut(block["sensors"], start, end)
    real = block["real"].reindex(sensors.index)
    rounds = block["rounds"][design]
    names = list(rounds["scores"])
    heights = [1]*5 + [1.35]*len(names) + [1.3]
    fig, axs = plt.subplots(len(heights), 1, figsize=(16, 18), sharex=True,
                            gridspec_kw={"height_ratios": heights})
    fig.subplots_adjust(left=.16, right=.98, top=.92, bottom=.055, hspace=.19)
    mode = "Ablação: score de nível/resíduo + C1 conservador exploratório" if design == "levels" else "Primeira montagem: score de nível + tendência"
    fig.suptitle(f"B8802B — {title}\n{mode}", fontsize=16, y=.984)
    fig.legend(handles=[
        Line2D([], [], color="#0369a1", lw=1, label="Sensor / score"),
        Line2D([], [], color="#dc2626", ls="--", label="Limiar do modelo"),
        Line2D([], [], marker="o", color="#d97706", ls="none", mfc="none", label="Acima do limiar"),
        Line2D([], [], marker="o", color="#dc2626", ls="none", label="Alarme: regra 15/20"),
        Line2D([], [], marker="|", color="#7c3aed", ls="none", markersize=12, label="Alerta: duração mínima 1 h"),
    ], loc="upper center", bbox_to_anchor=(.55, .951), ncol=5, fontsize=9, frameon=False)
    audit = []
    for ax, (col, label) in zip(axs[:5], SENSORS):
        series = with_gaps(sensors[col].where(real))
        ax.plot(series.index, series, color="#0369a1", lw=.7)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
    for row, name in enumerate(names):
        ax = axs[5+row]
        f = cut(rounds["scores"][name], start, end)
        threshold = rounds["thresholds"][name]
        # Only mask the displayed continuous line; stored alarm/alert flags are untouched.
        series = with_gaps(f.score.where(real))
        ax.plot(series.index, series, lw=.65, color="#0369a1")
        above = (f.score > threshold) & real
        ax.scatter(f.index[above], f.loc[above, "score"], s=9, edgecolor="#d97706", facecolor="none", linewidth=.55)
        ax.scatter(f.index[f.alarm], f.loc[f.alarm, "score"], s=9, color="#dc2626", zorder=4)
        ax.axhline(threshold, color="#dc2626", ls="--", lw=.8)
        # Symlog preserves zero and all peaks; each score has its own units and threshold.
        ax.set_yscale("symlog", linthresh=threshold, linscale=1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(MODELS[name].replace(" · ", "\n")+"\nscore (symlog)", fontsize=9)
        ax.text(.995, .93, f"limiar = {threshold:.6g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, zorder=6, bbox={"facecolor": "white", "alpha": .95, "edgecolor": "none"})
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
        axs[-1].plot(f.index[f.alert], np.full(int(f.alert.sum()), row), "|", color="#7c3aed", ms=10, mew=1)
        audit.append({"model": name, "above_markers": int(above.sum()),
                      "alarm_markers": int(f.alarm.sum()), "alert_markers": int(f.alert.sum())})
    axs[-1].set_yticks(range(len(names)), ["VAE", "C1", "C2", "C3", "C1 conserv."][:len(names)], fontsize=8)
    axs[-1].set_ylim(len(names)-.5, -.5)
    axs[-1].set_ylabel("Alerta\noperacional", fontsize=9)
    axs[-1].grid(alpha=.2)
    for ax in axs:
        ax.set_xlim(start, end)
        if start <= FAILURE < end:
            ax.axvline(FAILURE, color="#111827", ls=":", lw=1.2)
    if start <= FAILURE < end:
        axs[0].text(FAILURE, 1.02, "Falha conhecida · 06/jul/2022 10:00 ", ha="right",
                    transform=axs[0].get_xaxis_transform(), fontsize=9)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=11)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axs[-1].set_xlabel("Data/hora conforme os arquivos de origem · lacunas preservadas · sinais a cada 5 min")
    fig.text(.16, .018, "Sensores após filtros operacionais, antes do clipping/normalização. Scores em escalas próprias; symlog é linear até o limiar e log acima.\n"
             "Marcadores são os instantes salvos, sem interpolação. Alerta do modelo não equivale a falha confirmada. Pressões na unidade original do arquivo.", fontsize=9)
    return fig, pd.DataFrame(audit)


def make_notebook():
    nb = nbformat.v4.new_notebook()
    cells = [nbformat.v4.new_markdown_cell("# B8802B — sensores, scores e anomalias no tempo\n\n"
        "Comparação dos resultados já calculados, sem novo treino. As figuras mostram cinco sensores, um painel de score/limiar por modelo e uma faixa com os alertas operacionais.\n\n"
        "**Laranja:** ultrapassagem instantânea do limiar. **Vermelho:** alarme pela regra 15/20. **Roxo:** flag de alerta após a regra de duração mínima de 1 h. As cores não são rótulos de falha.\n\n"
        "O VAE é o bundle 2025. C1/C2/C3 abaixo usam a ablação de nível/resíduo; a primeira montagem aparece ao final. O limiar do C1 conservador foi escolhido exploratoriamente após examinar 2026. Os scores não compartilham unidade; cada painel mostra seu próprio limiar.\n\n"
        "Os gráficos preservam lacunas, usam sensores antes do clipping e exibem os flags originais. Os pontos de alarme/alerta **não são recalculados ao selecionar uma janela**. Os horários seguem os arquivos, sem conversão de fuso."),
        nbformat.v4.new_code_cell("from pathlib import Path\nfrom io import BytesIO\nimport sys\nfrom IPython.display import display, Image\nimport matplotlib.pyplot as plt\n\n"
            "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/b8802b_timeline_report.py').exists())\n"
            "sys.path.insert(0, str(ROOT / 'scripts'))\n"
            "from b8802b_timeline_report import load_artifacts, plot_window, window_summary\n"
            "data = load_artifacts()\n\n"
            "def show_figure(fig):\n    buffer = BytesIO()\n    fig.savefig(buffer, format='png', dpi=125, facecolor='white')\n    display(Image(data=buffer.getvalue()))\n    plt.close(fig)")]
    for slug, title, period, start, end, design, note in VIEWS:
        cells += [nbformat.v4.new_markdown_cell(f"## {title}\n\n{note}"),
                  nbformat.v4.new_code_cell(f"fig, audit = plot_window(data, {period!r}, {start!r}, {end!r}, design={design!r}, title={title!r})\nshow_figure(fig)\ndisplay(window_summary(data, {period!r}, {start!r}, {end!r}, design={design!r}))")]
    cells += [nbformat.v4.new_markdown_cell("## Explorar outra janela\n\nEdite as datas abaixo e execute a célula. Use `period='old'` para 2022 ou `period='recent'` para 2025–2026. `design='initial'` seleciona a primeira montagem; `design='levels'`, a ablação."),
              nbformat.v4.new_code_cell("# fig, audit = plot_window(data, 'recent', '2026-06-01', '2026-07-01', title='Junho de 2026')\n# show_figure(fig)")]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, NOTEBOOK)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_artifacts()
    body = ['<h1>B8802B — séries temporais e anomalias</h1>',
            '<p>Sensores, scores, limiares e alertas no mesmo eixo de tempo. Resultados originais preservados; não houve novo treino.</p>',
            '<p><b>Laranja:</b> acima do limiar. <b>Vermelho:</b> alarme 15/20. <b>Roxo:</b> alerta após duração mínima de 1 h. Alerta não é rótulo de falha. Lacunas não são interpoladas.</p>',
            '<p>O VAE usa o bundle 2025. Os primeiros gráficos mostram a ablação de nível/resíduo e o C1 conservador exploratório; a montagem inicial aparece ao final. A falha conhecida é marcada apenas em 2022. Os scores têm escalas próprias (symlog); as pressões mantêm a unidade do arquivo.</p>',
            '<p><a href="resultados.html">Abrir métricas e protocolo completo</a></p>',
            '<nav>'+' · '.join(f'<a href="#{v[0]}">{html.escape(v[1])}</a>' for v in VIEWS)+'</nav>']
    audits = []
    for slug, title, period, start, end, design, note in VIEWS:
        print(f"Plotando {slug}", flush=True)
        fig, audit = plot_window(data, period, start, end, design, title)
        path = OUT / f"serie_{slug}.png"
        fig.savefig(path, dpi=125, facecolor="white")
        plt.close(fig)
        summary = window_summary(data, period, start, end, design)
        # Check that plotted classifications and the tabulated saved flags agree.
        assert audit.alarm_markers.tolist() == summary["Alarme (pontos)"].tolist()
        assert audit.alert_markers.tolist() == summary["Alerta (pontos)"].tolist()
        assert audit.above_markers.tolist() == summary["Acima do limiar (pontos válidos)"].tolist()
        audit["view"] = slug
        audits.append(audit)
        summary.to_csv(OUT / f"serie_{slug}_resumo.csv", index=False)
        b64 = base64.b64encode(path.read_bytes()).decode()
        body += [f'<section id="{slug}"><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>',
                 f'<a href="serie_{slug}.png" target="_blank">Abrir figura em resolução completa</a>',
                 f'<img loading="lazy" alt="{html.escape(title)}: sensores e scores alinhados no tempo" src="data:image/png;base64,{b64}">',
                 '<details><summary>Contagens e primeiro alerta nesta janela</summary>'+summary.to_html(index=False, border=0, float_format=lambda x: f"{x:.6g}")+'</details></section>']
    pd.concat(audits).to_csv(OUT / "series_plot_audit.csv", index=False)
    style = "body{font:16px/1.6 system-ui;color:#172033;max-width:1500px;margin:28px auto;padding:0 24px}img{width:100%;height:auto}nav{padding:18px;background:#eef2ff}section{margin-top:50px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}a{color:#0369a1}details{overflow:auto}"
    (OUT / "series_temporais.html").write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B8802B — séries temporais</title><style>'+style+'</style><body>'+"\n".join(body)+'</body></html>')
    make_notebook()
    print(OUT / "series_temporais.html")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
