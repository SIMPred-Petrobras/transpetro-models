# Monitoramento e retreino — guia de integração

Este guia é para o time de integração. Ele cobre duas coisas:

- **A. O monitor semanal:** o que rodar, como ler e o que fazer com cada resultado. É o que precisa ser integrado.
- **B. O retreino:** o que acontece quando o normal do equipamento muda, quem faz cada passo e o que a integração
  recebe de volta.

O monitor **não muda nada na inferência**: é um passo a mais, depois dela, usando o CSV que ela já gera.

---

## A. O monitor semanal

### O que rodar (1× por semana, menos de 1 minuto)

```bash
# 1) a inferência normal (como já é feito) — gera <equip>_inferencia.csv
python3 B-8802B-2025/scripts/b8802b2025_exemplo.py

# 2) o monitor, sobre o resultado dela e o CSV bruto do mesmo período
python3 monitor_drift.py \
  --inferencia B-8802B-2025/scripts/b8802b2025_inferencia.csv \
  --alarm  B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE/alarm.json \
  --dados  B-8802B-2025/dados/2025_2026/data_2025-01-01_2026-08-10_raw.csv \
  --bundle B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE \
  --csv monitor_b8802b.csv --png monitor_b8802b.png
```

- `--dados` é o CSV bruto do período monitorado (o mesmo formato de entrada da inferência). Passe sempre
  `--dados` e `--bundle`: sem eles o monitor perde os indicadores M5 a M8, que são os mais importantes.
- O monitor usa dois arquivos de calibração que já vêm **dentro do bundle**: `drift_ref.json` (M6) e
  `residual_ref.json` (M8). Bundle sem um deles: o indicador correspondente é pulado.
- Dependências: as do `requirements.txt` (o monitor precisa de `scipy`, além de pandas e numpy).
- Guarde a saída de texto e o `--csv` de cada semana: o histórico é o que o time de modelos usa para investigar.

### Como ler a saída

A última parte diz o status:

| status | significado | o que a integração faz |
|---|---|---|
| 🟢 **VERDE** | modelo saudável | nada; arquivar a saída |
| 🟡 **AMARELO** | algo mudou de forma sustentada, ou há dado congelado | **avisar o time de modelos** com a saída do monitor; a inferência continua normalmente |
| 🔴 **VERMELHO** | o normal mudou de forma forte e sustentada: os alarmes do modelo perdem confiabilidade até a revisão | avisar o time de modelos **no mesmo dia**; tratar novos alarmes do modelo com ressalva até o retorno |

Junto com o status vem o motivo (qual indicador e, quando houver, **qual sensor**). Inclua-o no aviso.

**Amarelo ou vermelho não significam problema no equipamento.** Significam que o "normal" mudou em relação ao que
o modelo aprendeu (por exemplo, depois de uma manutenção). Problema no equipamento é o que o **alarme** da
inferência aponta. Por isso o monitor nunca silencia um alarme.

### O que o monitor mede

| indicador | o que mede | quando sai do verde |
|---|---|---|
| **M1 alarme %** | fração da semana com o modelo em alarme | o modelo passou a alarmar demais (amarelo > 0,5 % em 2 de 4 semanas; vermelho > 2 % em 4 de 6) |
| **M2 atenção %** | fração acima do nível de atenção | sinal precoce, só informativo |
| **M3 erro relativo** | quanto a operação atual está longe do normal aprendido (1× = igual ao treino) | > 2× (amarelo) ou > 2,5× (vermelho) em 6 de 8 semanas |
| **M4 cobertura** | quantidade de dado válido na semana | semanas com menos de 1 dia de operação não contam |
| **M5 fora da faixa** | % do tempo com um sensor fora da faixa do treino (ali o modelo fica "cego" a ele) | > 10 % (amarelo) ou > 25 % (vermelho) |
| **M6 KS diário** | compara a distribuição de cada dia com a do treino, sensor a sensor; dispara com 3 de 5 dias diferentes | qualquer disparo nas últimas 4 semanas → amarelo; diz **via qual sensor** |
| **M7 dado congelado** | horas da semana com 3 ou mais sensores com o mesmo valor por 12 h ou mais (falha de aquisição) | qualquer hora → amarelo; **avisar também a instrumentação** |
| **M8 temperatura de mancal** | nível da temperatura descontada a estação (resíduo de um modelo de comportamento normal) | 3 de 5 dias fora da faixa normal nas últimas 4 semanas → amarelo; diz qual temperatura |

Por que M7 e M8 existem:

- **M7:** durante o dado congelado os valores parecem normais, então o modelo de anomalia **não percebe nada**.
  Uma falha nesse período passaria batida. Esses trechos também ficam fora do M6, para não virarem falsa mudança.
- **M8:** o M6 compara um dia com o ano inteiro de treino. Em temperatura, que tem estação forte, isso o deixa
  pouco sensível. No B-8802B, a temperatura do mancal LA caiu ~6 °C em 28/03/2026 e o M6 não disparou; o M8
  disparou em 4 dias. Ele foi validado também no B-4064A (achou a mudança pós-reparo em ~2,5 dias, sem nenhum
  disparo falso). O M8 só é configurado para temperaturas que a regressão explica bem: no B-8802B, só o mancal LA.

### O que mandar no aviso

1. A saída de texto do monitor (status, motivos, tabela das últimas semanas) e a figura.
2. **Eventos do período**, se houver: manutenções ou intervenções no equipamento (com data e horário),
   mudanças de faixa de operação, trocas ou recalibrações de sensor, falhas de comunicação. Esses eventos mudam a
   interpretação: o mesmo sinal pode ser um normal novo, um sensor trocado ou uma degradação.

---

## B. Quando o normal muda: o retreino

### O princípio

O retreino **não tem data marcada** e **nunca é automático**. O modelo em produção fica fixo; ele só é retreinado
quando o normal do equipamento mudou de verdade **e a operação confirmou**. Retreinar sozinho a cada aviso seria
perigoso: o monitor também reage a uma degradação real, e retreinar nela colocaria a falha dentro do "normal".

### O fluxo

| passo | quem | o que acontece |
|---|---|---|
| 1. Aviso | integração | o monitor sai do verde; a integração avisa o time de modelos (seção A) |
| 2. Investigação | time de modelos + operação | o time de modelos cruza os sinais (qual indicador, qual sensor, desde quando) e pergunta à operação o que aconteceu no período |
| 3. Decisão | operação confirma, time de modelos decide | **degradação:** não retreina, os alarmes valem e são o que interessa · **sensor trocado ou recalibrado:** só recalibrar (sem retreino) · **normal novo** (manutenção, regime de operação): retreinar |
| 4. Coleta | integração | o modelo atual continua rodando; os alarmes dele passam a ser tratados com ressalva. A integração garante que o dado bruto do período novo está disponível para o time de modelos |
| 5. Retreino | time de modelos | com o dado desde a mudança (detalhe abaixo); cada modelo novo passa pela bateria de validação antes de ser entregue |
| 6. Entrega | time de modelos | um **bundle novo em pasta nova**, ao lado do atual, já com `drift_ref.json` e `residual_ref.json` calibrados no período de treino dele |
| 7. Sombra | integração | rodar **os dois bundles por 4 semanas** (inferência e monitor), sem trocar nada |
| 8. Troca | integração, com o aval do time de modelos | trocar os caminhos para o bundle novo quando ele estiver verde no monitor e as diferenças de alarme entre os dois tiverem sido entendidas. O bundle antigo fica guardado para voltar atrás |

### O retreino em si (passo 5)

- **Modelo definitivo:** com **12 meses** de dado do normal novo (ou ≥ 4000 h de operação cobrindo os regimes do
  ano). É o caso do B-8802B-2025, treinado com 2025 inteiro. Roda pelo pipeline com portão
  (`scripts/retrain_pipeline.py --operacao-confirmou`), que só aceita janela confirmada pela operação e entrega o
  bundle aprovado pela bateria (falso positivo fora do treino, falha real de 2022, falha simulada).
- **Antes dos 12 meses (em validação):** a partir de 1 mês de dado pode entrar um **modelo provisório**,
  retreinado todo mês com tudo o que acumulou, até virar o definitivo. No teste do B-8802B ele quase não dá alarme
  falso desde o 1º mês, mas só detecta falha de forma confiável a partir de ~8 meses. Os alertas de um provisório
  valem **com ressalva**. Para a integração, isso significa receber um bundle novo por mês nesse período, cada um
  com o mesmo passo de sombra.
- **Os detectores andam junto com o modelo.** Todo bundle novo vem com `drift_ref.json` (M6) e `residual_ref.json`
  (M8) calibrados no **mesmo período** do treino do modelo, porque o monitor responde se aquele modelo ainda descreve
  o equipamento. A referência não é uma janela deslizante: uma janela que acompanha os últimos meses absorveria a
  própria mudança (no B-8802B, a mudança de 2026 teria virado "normal" em um mês, sem ninguém confirmar).

### Revisão anual

Uma vez por ano, mesmo com tudo verde, o time de modelos repete a bateria de validação com o modelo em produção.
Sem um aviso confirmado, não se retreina.

---

## Referência para o time de modelos

Calibrar os detectores de um bundle (a referência padrão é a janela de treino gravada no `alarm.json`):

```bash
# M6 (KS): quantis da referência, sem dado congelado
python3 monitor_drift.py --make-drift-ref --dados <CSV bruto> --bundle <pasta do bundle>

# M8 (temperatura de mancal descontada a estação): avisa se a regressão explica pouco o sensor (R² < 0,5)
python3 monitor_drift.py --make-residual-ref --dados <CSV bruto> --bundle <pasta do bundle> \
  --alvos "Temperatura Bomba LA" \
  --preditores "Temperatura Motor LA,Temperatura Motor LNA,Pressão Descarga,Pressão Sucção"
```

`scripts/retrain_pipeline.py` faz as duas calibrações sozinho em todo bundle novo (o M8 herda alvos e preditores do
bundle em produção). Política completa: `docs/politica_retreino.md`. Análise que embasa o M7 e o M8:
`notebooks/drift/mudanca_conceito_B-8802B.ipynb`.

## Situação atual do B-8802B

- **Status: AMARELO pelo M8** desde 01/04/2026: a temperatura do mancal LA está ~6 °C abaixo do esperado desde
  28/03/2026, logo depois de uma parada da bomba. O M6 não dispara. Pergunta em aberto à operação: o que foi feito
  nas paradas de 17/03 (~00:30 às 09:30) e de 27/03/2026 (~08:30 às 14:30)?
- **Dado congelado** (M7) em 02/01/2025, 15–23/05/2025 e 22/02–04/03/2026: vale a instrumentação verificar.
