# Política de monitoramento de drift e retreino — SIMPred

**Escopo.** Modelos de detecção de anomalia (autoencoders, erro de reconstrução + régua μ+y·σ) em
produção. Escrita a partir do caso B-8802B (modelo de 2022 lendo 2025–26; retreino B-8802B-2025;
walk-forward; experimento de feature engineering), com números calibrados nesse equipamento. Vale como
padrão para os demais, ajustando os limiares quando houver histórico de produção deles.

Ferramenta: `scripts/monitor_drift.py` (só pandas/numpy; lê o CSV que o pacote de deploy já gera).

---

## 1. Princípio: modelo congelado, retreino por gatilho — não por calendário

Um modelo não-supervisionado aprende "o normal". Ele erra por dois motivos opostos, e a política
precisa distinguir os dois:

| | o que acontece | o que fazer |
|---|---|---|
| **Concept drift** | o *normal* mudou (reparo, novo regime de operação, sensor trocado) e o modelo passa a estranhar a operação sã | **retreinar** (ou recalibrar) |
| **Degradação** | o equipamento está saindo do normal — é justamente o que o modelo existe para pegar | **alarmar** — e **nunca** retreinar em cima disso |

Por que não retreinar por calendário (ex.: a cada 3 meses):
- **Absorve degradação lenta.** Um mancal que esquenta ao longo de semanas vira "novo normal" a cada
  retreino e o modelo nunca alarma.
- **Janelas curtas perdem sensibilidade.** No walk-forward do B-8802B, janelas de 3 e 6 meses
  **não detectam a falha sintética**; só a de 12 meses detecta. Menos alarmes ali não era modelo
  melhor — era modelo mais cego.
- **Não protege contra regime novo.** A janela jan–jun/25 ficou estável por 8 meses e disparou 12 %
  de alarme em abr/26 quando apareceu um regime que ela não viu. Calendário não antecipa isso;
  monitoramento sim.

O que se faz em vez disso: o modelo fica **congelado e monitorado** semanalmente; retreina-se quando
um gatilho dispara **e** a checagem drift × degradação confirma que é drift.

## 2. O que monitorar (semanal, só tempo de operação)

Calculado sobre a saída do deploy (`<equip>_inferencia.csv`: erro de reconstrução, alarme,
severidade), agregando por semana e ignorando semanas com menos de 1 dia de operação (288 instantes
de 5 min).

| indicador | definição | o que revela |
|---|---|---|
| **M1 alarme %** | fração dos instantes da semana em alarme (após persistência 15-de-20) | o gatilho principal; drift faz isso subir e **ficar** alto |
| **M2 atenção %** | fração acima do nível de atenção (μ+4σ) | sinal precoce, mais ruidoso |
| **M3 erro relativo** | mediana do erro na semana ÷ μ do treino (`alarm.json → threshold_calibration.mean_normal`) | quanto o "normal atual" está longe do "normal aprendido" — sobe **antes** do alarme |
| **M4 cobertura** | instantes válidos na semana (esperado ≈ 2016 a 5 min) | qualidade de dado / sensor fora; uma semana vazia não é "verde" |
| **M5 saturação do clip** | fração dos instantes em que um sensor está **fora da faixa de clip do treino** (p1–p99,9); reporta-se o pior sensor | drift **por omissão**: fora da faixa o valor é truncado e o modelo deixa de ver o sensor — o alarme não cai, a **sensibilidade** cai. Exige o CSV bruto (`--dados`) |
| **M6 KS diário por sensor** | teste de Kolmogorov–Smirnov de cada dia de operação contra a amostra de referência do treino, por sensor; limiar do estatístico D calibrado na própria referência; dispara quando **3 dos últimos 5 dias** ficam acima do limiar | o detector mais rápido de mudança de conceito (~4 dias no drift real do B-8802B) e diz **via qual sensor** mudou; calibração guardada no `drift_ref.json` do bundle |

**Valores de referência medidos no B-8802B (84 semanas, jan/25 → ago/26):**

| | modelo saudável (B-8802B-2025) | modelo com drift (modelo 2022 lendo 2025–26) |
|---|---|---|
| alarme % — mediana semanal | 0,00 % | **7,2 %** |
| alarme % — máximo semanal | 1,0 % (2 de 84 semanas > 0,5 %, nunca 2 seguidas) | 56,7 % (65 de 84 semanas > 2 %) |
| erro relativo — mediana | 0,98× (≈ 2× em jul/26 — acompanhar) | **3,2×** |
| atenção % — mediana | 0,4 % | 36 % |
| M5 — Temp. mancal LA fora da faixa | 0–2 % em 2025 → **17–19 % em 2026-T2/T3** (piso 43,9 °C) | — |

A separação entre os dois casos é de uma ordem de grandeza — os gatilhos abaixo cabem folgados entre eles.

## 3. Gatilhos (semáforo) — regra k-de-n sobre as últimas semanas

Regra "k das últimas n semanas" em vez de "n semanas seguidas": uma semana quieta no meio de um drift
(parada, pouca operação) não zera a contagem.

| nível | dispara quando… | ação |
|---|---|---|
| 🟢 **Verde** | nenhuma regra abaixo | nada |
| 🟡 **Amarelo** | alarme % > 0,5 em **≥ 2 das últimas 4** semanas **ou** erro relativo > 2,0 em **≥ 6 das últimas 8** **ou** M5 > 10 % em **≥ 2 das últimas 4** | **investigar**, não retreinar: houve manutenção? mudou a faixa de operação? sensor trocado/recalibrado? há assinatura física (seção 4)? |
| 🔴 **Vermelho** | alarme % > 2 em **≥ 4 das últimas 6** semanas **ou** erro relativo > 2,5 em **≥ 6 das últimas 8** **ou** M5 > 25 % em **≥ 4 das últimas 6** | aplicar o checklist da seção 4; confirmado drift → abrir retreino (seção 5) |

Validação da regra nos dois casos reais: modelo saudável → **84 de 84 semanas verdes**; modelo com
drift → amarelo na 2ª semana, **vermelho na 4ª** (26/01/2025) e 67 de 84 semanas vermelhas. Ou seja,
o drift do B-8802B teria sido sinalizado em **um mês**, não em 20. Com M5 ligado, o modelo de produção
mostra **13 semanas amarelas desde 26/04/2026** (temperatura do mancal LA fora da faixa aprendida), 0 vermelhas —
o sinal precoce que os indicadores de alarme não veem.

**Validação numérica completa** (índice de mudança de conceito por trimestre, falha sintética injetada mês a
mês, saturação do clip, teste de estresse): `results/monitor_drift/validacao_politica_retreino_b8802b.html`
(gerado por `validacao_drift.py` + `build_validacao_html.py` na mesma pasta). Resumo: 2022→2025 = **3,4×**
o normal do modelo 2022; 2025→2026-T3 = **2,2×** o normal do modelo atual (1,1× → 1,9× → 2,2×, crescendo);
antecedência da sintética 100 % **20 h em 2025 → 19 h em 2026** (sensibilidade mantida até aqui); FP robusto
a −15 °C / +6 bar (< 0,5 %), mas −3 °C adicionais já atrasam a sintética para 14 h **depois** do fim da rampa —
o drift atual erra por omissão, não por excesso.

**Gatilhos por evento** (independem do semáforo — a integração/operação avisa):
- **Intervenção no equipamento** (reparo, troca de mancal/acoplamento/selo, alinhamento): o modelo entra
  em **observação por 30 dias**. Alarme sustentado logo após a intervenção é esperado (B-4064A: +20 °C
  no mancal pós-reparo virou um ano de falso positivo). Se após 30 dias de operação estável o semáforo
  continua amarelo/vermelho → retreino com o novo normal.
- **Mudança operacional declarada** (nova faixa de pressão/vazão, novo produto): mesmo tratamento.
- **Sensor trocado, recalibrado ou realocado**: verificar M4 e a escala do sensor antes de qualquer
  conclusão; um degrau de sensor não é nem drift nem degradação.

## 4. Checklist drift × degradação (antes de qualquer retreino)

Retreinar sobre uma janela com degradação **apaga a falha do modelo**. Antes de abrir retreino,
responder:

1. **Há assinatura física?** Vibração e/ou temperatura de mancal com **tendência monotônica** (subindo
   dia após dia), erro concentrado nesses sensores, alarme cada vez mais frequente/longo → é
   **degradação**: tratar como alarme para a operação, **não retreinar**.
2. **O alarme é intermitente e acompanha o regime?** Picos quando a pressão muda de patamar, semanas
   quietas alternadas com semanas ruidosas, sensores oscilando mas sem tendência → padrão de **drift**.
3. **Houve evento explicativo?** Manutenção, mudança operacional, troca de sensor (seção 3).
4. **A operação confirma que o período foi normal?** Sem essa confirmação por escrito, o período
   não entra em janela de treino.
5. **Recalibrar a régua resolve?** Só se o erro relativo estiver **≤ 1,5×**: aí o normal mudou pouco
   e basta refazer μ e σ na janela nova (`scripts/recalibrate_threshold.py --method sigma`) e revalidar
   (seção 5, passo 5). Acima disso, recalibrar mata a sensibilidade — no B-8802B, re-baselinar o modelo
   de 2022 daria limiar 2,86, **acima do pico da própria falha de 2022 (~2,5)**: a falha passaria batida.
   → **retreinar**.

## 5. Procedimento de retreino (o que foi feito no B-8802B-2025 e passa a ser o padrão)

1. **Janela de treino ≥ 12 meses de operação** (cobrir todos os regimes do ano), ou desde a intervenção
   se houver **≥ 4000 h de operação** e os regimes estiverem representados. Excluir períodos sob suspeita
   (checklist) e os 90 min após partidas e degraus de processo (máscara já no pipeline).
2. **Held-out temporal** que o treino não vê (≥ 3–5 meses) para medir falso positivo honestamente.
3. **AutoML + seleção por FP held-out**, nunca pelo seletor automático quando não há falha na janela
   (ele maximiza "alerta pré-falha", que sem falha é só o FP da última semana). Rodar **≥ 3 seeds**:
   1 em 3 seeds do B-8802B falhou a restrição de FP só pela inicialização.
4. **Bateria de sensibilidade** obrigatória — sem ela, "menos alarme" é indistinguível de "mais cego":
   falha(s) real(is) do histórico do equipamento pontuadas com o modelo novo (cross-era), falha sintética
   com a assinatura medida (100 % e 50 %), e os episódios de prioridade alta do período recente que
   devem sobreviver (âncoras).
5. **Régua μ + y·σ** na janela normal do treino (y por equipamento; B-8802B: 6,5) e persistência
   15-de-20; gravar tudo em `alarm.json` (`scripts/recalibrate_threshold.py --method sigma`).
6. **Bundle novo em pasta própria**, lado a lado com o anterior (`B-8802B-2025` ao lado de `B-8802B`);
   o antigo permanece para comparação e rollback. Guia de migração para a integração (`MIGRACAO.md`).
7. **Sombra por 4 semanas**: os dois modelos rodam em paralelo; troca-se quando o novo fica verde no
   monitor e os alarmes divergentes foram entendidos.

## 6. Revisão anual (mesmo tudo verde)

Uma vez por ano, por equipamento: rodar o monitor sobre os 12 meses, repetir a bateria de sensibilidade
com o modelo em produção (a falha sintética continua sendo detectada?), verificar se surgiu regime novo
(distribuição de pressão/carga vs a do treino) e registrar. Sem gatilho, **não** se retreina — a revisão
existe para pegar deriva lenta demais para o semáforo semanal.

## 7. Papéis

| quem | faz |
|---|---|
| **Integração** | roda o deploy e o `monitor_drift.py` semanalmente; comunica intervenções/mudanças operacionais/troca de sensor |
| **Ciência de dados** | atua em amarelo/vermelho (checklist), conduz retreino e bateria, entrega bundle novo + migração |
| **Operação/manutenção** | confirma eventos e se o período foi normal (pré-requisito para janela de treino); responde aos alarmes |

## 8. Como rodar o monitor

```bash
python scripts/monitor_drift.py \
  --inferencia deploy_v2/Transpetro/B-8802B-2025/scripts/b8802b2025_inferencia.csv \
  --alarm deploy_v2/Transpetro/B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE/alarm.json \
  --dados deploy_v2/Transpetro/B-8802B-2025/dados/2025_2026/data_2025-01-01_2026-08-10_raw.csv \
  --bundle deploy_v2/Transpetro/B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE \
  --png monitor_b8802b.png --csv monitor_b8802b.csv
```

`--dados` + `--bundle` são opcionais e habilitam M5 (saturação do clip); reutilizam o `simpred_inference.py`
do pacote de deploy (pasta `Transpetro/`, dois níveis acima do bundle) para aplicar os mesmos passos temporais.

Imprime a tabela das últimas semanas, o status (verde/amarelo/vermelho) com a regra que disparou, e o
histórico do semáforo; `--png` gera a figura (alarme % e erro relativo por semana, semanas não-verdes
sombreadas). O script não depende da lib interna e pode ser copiado para o pacote de deploy.

## 9. Pendências conhecidas

- Limiares calibrados só no B-8802B; para B-6511502A e B-3403C, rodar o monitor sobre o histórico de
  produção assim que houver e ajustar se necessário (a estrutura da regra é a mesma).
- **B-8802B-2025 está em deriva mensurável**: erro relativo 2,2× em 2026-T3 e temperatura do mancal LA
  (mediana 55 → 46 °C) fora da faixa aprendida 17–19 % do tempo (M5 amarelo desde 26/04/2026). FP e
  sensibilidade ainda intactos. Ação: (a) perguntar à operação a causa do resfriamento (ambiente? produto?
  sensor?); (b) deixar preparado o retreino com janela jan/25 → ago/26 (mais recente) para disparar quando
  erro relativo > 2,5× ou M5 > 25 %, ou na revisão anual — o que vier primeiro.
- Se o preset `detrend7d` for validado pela operação (ver `fe_experiment_resultado.ipynb`), repetir o
  walk-forward com ele: a hipótese é que torne janelas curtas menos frágeis a regime novo.
