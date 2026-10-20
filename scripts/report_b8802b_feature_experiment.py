"""Generate a self-contained research report from completed local experiments."""
import base64
import html
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/b8802b_feature_candidates"
B = ROOT / "results/b8802b_feature_candidates_levels"
OUT = ROOT / "reports/b8802b_candidate_results"
NAMES = ["VAE_atual", "C1_vibracao", "C2_pressao", "C3_dinamica_termica"]
LABELS = ["VAE atual", "C1: vibração", "C2: + pressões", "C3: + dinâmica térmica"]


def fmt(x):
    return "—" if pd.isna(x) else f"{x:.2f}".replace(".", ",")


def comparison(folder):
    h = pd.read_csv(folder / "history.csv")
    ev = pd.read_csv(folder / "events.csv").set_index("model")
    s = pd.read_csv(folder / "synthetic.csv")
    rows = []
    for name, label in zip(NAMES, LABELS):
        recent = h[h.model.eq(name)&h.period.isin(["jan-mai2026", "jun-ago2026"])]
        g = s[s.model.eq(name)&s.date.isin(["2026-06-10", "2026-07-15"])&s["shape"].eq("ramp48")&~s.thermal_only&s.scale.eq(.5)]
        assert len(g) == 6 and g.common_eligible.all()
        rows.append({"Modelo": label, "Episódios 2026": int(recent.episodes.sum()),
            "Horas em alerta": fmt(recent.alert_points.sum()/12),
            "Antecedência 2022 (h)": fmt(ev.loc[name, "historical_lead_h"]),
            "Evento 17/jan": "sim" if ev.loc[name, "january_anchor_detected"] else "não",
            "Rampa 50%: detecta / antes do fim": f"{g.lead_hours.notna().sum()}/6 · {g.lead_hours.ge(0).sum()}/6"})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary, ablation = comparison(A), comparison(B)
    primary.to_csv(OUT / "primary_summary.csv", index=False)
    ablation.to_csv(OUT / "ablation_summary.csv", index=False)
    trade = pd.read_csv(B / "threshold_tradeoff_exploratory.csv")
    c = trade[trade.model.eq("C1_vibracao") & trade.q.eq(1)].iloc[0]
    episodes = pd.read_csv(B / "C1_conservative_exploratory_episodes.csv")
    with (B / "candidates.pkl").open("rb") as f: model = pickle.load(f)
    physical_limits = model["base_norm"][0]+c.threshold*model["base_norm"][1]

    syn = pd.read_csv(B / "synthetic.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, date, month in zip(axs, ["2026-06-10", "2026-07-15"], ["Junho", "Julho"]):
        w = syn[syn.model.eq("VAE_atual") & syn.date.eq(date) & syn["shape"].eq("ramp48") & ~syn.thermal_only & syn.scale.eq(.5)]
        ax.plot(w.temperature, w.lead_hours, "o-", color="#64748b", label="VAE atual")
        missing = w[w.lead_hours.isna()]
        if len(missing):
            ax.scatter(missing.temperature, [-5]*len(missing), marker="x", color="#64748b")
            ax.text(.02, .02, "× = sem detecção (posição ilustrativa)", transform=ax.transAxes, fontsize=8)
        val = c[f"{date}_0.5_lead_h"]
        ax.plot([40, 50, 60], [val]*3, "o--", color="#0284c7", label="C1 conservador (exploratório)")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{month}: mesma rampa de vibração a 50%", xlabel="Temperatura imposta (°C)",
            ylabel="Antecedência ao fim da rampa (h)", xticks=[40, 50, 60])
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle("Estabilidade térmica recuperada; resultado exploratório, sem aprovação para deploy")
    fig.tight_layout()
    fig.savefig(OUT / "thermal_robustness.png", dpi=170)
    plt.close(fig)

    sections = []
    def add(title, text, table=None): sections.append((title, text, table))
    add("Resultado e decisão", "Executamos os três candidatos e uma ablação de engenharia de features. A primeira montagem preservou a falha forte, mas perdeu rampas menores. A ablação recuperou sensibilidade com mais alertas. Nenhum candidato cumpriu simultaneamente todos os critérios para substituir o VAE. O resultado mais promissor para avaliação em paralelo foi um detector simples de vibração com limiar conservador, identificado na análise exploratória posterior. Produção e notebooks existentes não foram alterados.")
    add("Protocolo", "B8802B: trinca nas lâminas do acoplamento em 06/07/2022 às 10:00. Ajuste dos candidatos em jan–set/2025 (60.667 pontos operacionais); calibração em out–dez/2025 (21.992). Avaliação em jan–10/ago/2026 e replay de 2022. O VAE é o bundle B-8802B-2025 existente, treinado com 2025, não o bundle antigo de 2022. Não refizemos seu treino. Os dados recentes não possuem rótulos completos: os números abaixo são episódios de alerta, não falsos positivos confirmados.")
    add("Definição dos candidatos", "C1 usa medianas de 30 minutos das vibrações LA/LNA, padronizadas por centro/dispersão robustos de treino. A montagem inicial acrescenta o contraste entre medianas de 1h e 6h. C2 acrescenta resíduos de regressões de vibração condicionadas às pressões atuais e passadas. C3 acrescenta, aos preditores de C2, variações de temperatura em 30min/2h e desvio da mediana térmica causal de 6h. Nenhum recebe nível térmico absoluto. C2 e C3 preservam o canal C1 em paralelo. Os regressores são histogram gradient boosting de mediana: 150 iterações, 15 folhas, mínimo 150 amostras por folha, learning rate 0,05, sem early stopping aleatório. Não recebem a própria vibração como entrada.")
    add("Comparabilidade e limiares", "Todos usam a mesma regra do deploy: 15 de 20 leituras válidas de 5min em 100min, seguida da regra de duração mínima de uma hora; episódios separados por mais de uma hora. O limiar de cada candidato é o menor de uma grade fixa de quantis que gera zero alertas na calibração. O VAE também tem zero nessa janela. Essa igualdade na calibração não garante igualdade de alertas em 2026. Os candidatos não achatam vibração por clipping superior, ao contrário do VAE; a comparação avalia pipelines completos, não isola causalmente uma única alteração. Janelas dos candidatos são causais e reiniciam em gaps/paradas.")
    add("Primeira montagem: nível mais tendência no score", "A tendência normalizada dominou o limiar de calibração, que ficou em 10,342 nos três candidatos. Embora tenham preservado aproximadamente 49,5h de antecedência na falha forte, perderam o evento de janeiro e as rampas a 50%. Foram rejeitados. Aumentar sensibilidade mediante simples redução do limiar conflita com a carga de alertas dessa montagem.", primary)
    add("Ablação posterior: retirar tendência do score", "Mantivemos dados, regressores, persistência e política de calibração. Retiramos somente os canais de contraste temporal do score, preservando medianas de 30min dos níveis/resíduos. Os candidatos recuperaram o evento de janeiro e as rampas menores. Os seis cenários são duas trajetórias (junho e julho), cada uma repetida a 40/50/60 °C; não são seis falhas independentes. A primeira contagem é detecção até o fim do platô de 24h; a segunda é antes do fim da rampa de 48h.", ablation)
    add("Cuidado com a antecedência de C3", "As 114,5h de C3 na ablação correspondem ao primeiro alerta dentro da janela pré-falha, em 01/jul/2022. C3 também alertou no período anterior presumido normal, por 1,05% dos instantes operacionais. Logo, esse número não comprova antecipação útil da falha. C2 apresentou a mesma taxa pré-rampa; C1 teve zero e o VAE aproximadamente 0,15%. C3 não apresentou benefício suficiente para justificar sua complexidade nesta rodada.")
    add("C1 conservador: ponto exploratório promissor", f"Examinando a grade de limiares já calculada na calibração, o C1 no máximo da referência (q=1, score {c.threshold:.6f}) teve 2 episódios e 6,25h em alerta em 2026, contra 1 episódio e 0,25h do VAE. Preservou 49,75h de antecedência na falha real e o evento de 17/jan. Nas rampas de 50%, alertou {fmt(c['2026-06-10_0.5_lead_h'])}h antes do fim em junho e {fmt(c['2026-07-15_0.5_lead_h'])}h em julho, independentemente do patamar térmico. As versões sem injeção não alertaram nessas duas janelas. Esse ponto foi identificado depois de examinar 2026: é resultado exploratório, não teste final independente. Um limiar baseado no máximo também depende de extremos da calibração e precisa ser verificado em outros blocos temporais.")
    add("Interpretação física do C1", f"Nesse ponto, os limiares equivalem aproximadamente a mediana de 30min da vibração LA acima de {fmt(physical_limits.iloc[0])} mm/s OU LNA acima de {fmt(physical_limits.iloc[1])} mm/s, seguida da persistência do deploy. São limiares empíricos desta referência, não limites normativos de segurança. A temperatura não decide o alerta. Os dois episódios foram 11/jan (adicional ao VAE) e 17/jan (já identificado pelo VAE). No episódio adicional, as medianas de vibração foram aproximadamente 3,28 e 1,46 mm/s; temperatura 49,4 °C. É necessário classificar esse evento com a operação antes de chamá-lo de falso positivo ou ganho de detecção.", episodes[["start", "end", "alert_hours"]].rename(columns={"start": "Início", "end": "Fim", "alert_hours": "Horas com flag de alerta"}))
    add("Sensibilidade térmica", "Nos três candidatos, os testes de patamar constante a 40/50/60 °C e de deslocamento de −10/0/+10 °C preservando a dinâmica mantiveram o score dentro da tolerância numérica de 1e−8. Quantizamos as features térmicas a 0,001 °C para evitar que arredondamentos de ponto flutuante mudem o ramo de uma árvore. No VAE, 40 °C é clippado para aproximadamente 43,88 °C. As intervenções são testes artificiais de sensibilidade; não validam todas as combinações como fisicamente realizáveis. A invariância aqui é de nível/offset, não a toda mudança possível de regime ou aquisição.")
    add("Cobertura e bateria", "Foram executadas 480 combinações por rodada: cinco datas de rampas de 48h, duas datas de degraus e rampas de 168h, controles sem injeção, amplitudes 25/50/100%, três temperaturas e quatro modelos, além de sondagens exclusivamente térmicas. Março/abril tinham pouca cobertura e algumas condições de janeiro já alertavam no controle; a elegibilidade exige pelo menos 80% de cobertura processada e controles limpos para todos os modelos comparados. A tabela principal usa somente junho/julho para manter o mesmo denominador entre rodadas. Os testes térmicos auxiliares acrescentaram 10 °C às bases de 40/50/60 °C; a variante de 60→70 °C extrapola a faixa confirmada e não fundamenta a aprovação. Nas sondagens de degrau, antecedência negativa significa atraso após o degrau, não erro de cálculo.")
    add("Limitações e próximos passos", "O histórico e as assinaturas já foram explorados anteriormente; a ablação e a curva de limiares desta rodada também foram motivadas pelos resultados. Não há estimativa independente de generalização, nem eventos suficientes para afirmar 100% de recall. Nenhum candidato reduziu simultaneamente a carga de alerta e preservou todos os critérios de sensibilidade. Recomendo levar o C1 conservador para avaliação em paralelo, classificar o episódio de 11/jan e testar calibração em múltiplos blocos/regimes e dados posteriores a agosto. C2 permanece como alternativa a desenvolver; acrescentar a dinâmica térmica não demonstrou ganho decisivo. Não substituímos o modelo de produção.")
    add("Correção do AutoML e validação", "Adicionamos failure_observed=False às configurações B-8802B-2025 e FE, herdado pelas variantes dT. O runner passa a ordenar por menor taxa held-out quando a data é sentinela, sem premiar pseudo-detecção pré-falha e sem fallback a métricas de treino ausentes. A bateria de sensibilidade externa continua necessária: um modelo silencioso não é aprovado por esse ranking. Cinco testes de regressão passaram. Nas duas execuções também passaram verificações de causalidade, reset em gaps, invariância térmica, paridade exata de alarme/alerta com o deploy e recarga dos artefatos.")
    add("Reprodução e arquivos", "Rodada inicial: OMP_NUM_THREADS=2 .venv/bin/python scripts/b8802b_feature_experiment.py. Ablação: adicionar --score-design levels --out results/b8802b_feature_candidates_levels. Curva posterior: .venv/bin/python scripts/b8802b_threshold_tradeoff.py. Relatório: .venv/bin/python scripts/report_b8802b_feature_experiment.py. Testes: .venv/bin/python -m unittest discover -s tests -p test_b8802b_selection.py -v. Artefatos completos, thresholds, modelos, scores por instante, casos sintéticos e hashes estão em results/b8802b_feature_candidates e results/b8802b_feature_candidates_levels. Os diretórios results/ e reports/ são ignorados pelo Git; scripts e testes ficam versionáveis.")

    # Assert report counts against the artifacts, not a manually assumed total.
    count = len(pd.read_csv(A / "synthetic.csv"))
    sections = [(title, text.replace("480 combinações", f"{count} combinações"), table) for title, text, table in sections]
    md = ["# B8802B — resultados dos candidatos\n", "Execução local em 17/10/2026.\n",
          "[Ver séries temporais: sensores, scores, limiares e alertas](series_temporais.html)\n"]
    body = ["<h1>B8802B — resultados dos candidatos</h1><p>Execução local em 17/10/2026.</p>",
            '<p><a href="series_temporais.html">Ver séries temporais: sensores, scores, limiares e alertas</a></p>']
    for title, text, table in sections:
        md.extend([f"## {title}\n", text+"\n"])
        body.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>")
        if table is not None:
            md += ["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"]*len(table.columns)) + " |"]
            md += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None)]
            md += [""]
            body.append(table.to_html(index=False, border=0))
        if title == "Sensibilidade térmica":
            md += ["![Sensibilidade térmica](thermal_robustness.png)\n"]
            b64 = base64.b64encode((OUT / "thermal_robustness.png").read_bytes()).decode()
            body.append(f'<img alt="Sensibilidade térmica" src="data:image/png;base64,{b64}">')
    (OUT / "resultados.md").write_text("\n".join(md))
    style = "body{font:16px/1.6 system-ui,sans-serif;color:#172033;background:#f8fafc;margin:40px auto;max-width:1120px;padding:0 24px}h1,h2{color:#0f172a}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border-bottom:1px solid #cbd5e1;text-align:left;padding:10px}th{background:#e2e8f0}img{max-width:100%;margin:20px 0}"
    (OUT / "resultados.html").write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B8802B — comparação de candidatos</title><style>'+style+'</style><body>'+"\n".join(body)+"</body></html>")
    print(OUT / "resultados.html")


if __name__ == "__main__":
    main()
