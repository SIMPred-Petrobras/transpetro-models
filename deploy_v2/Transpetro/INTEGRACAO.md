# Guia de integração — anomalia, monitor de mudança e retreino

Este é o guia principal para o time de engenharia. Ele explica **o que integrar, como, com que frequência, o que
cada peça precisa e o que ela entrega**, e responde às perguntas mais comuns sobre o retreino (requisitos,
desempenho, AutoML). Os detalhes de cada peça ficam nos guias específicos, citados ao longo do texto:

- `README.md`: o pacote de inferência (estrutura, dependências, contrato).
- `B-8802B-2025/MIGRACAO.md`: a troca do modelo de 2022 pelo de 2025 (validação e rollback).
- `MONITORAMENTO.md`: o monitor semanal em detalhe (indicadores M1 a M8, como ler cada status).

Exemplo usado em todo o guia: **B-8802B**, modelo em produção `B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE`.

---

## 1. A ideia em um minuto

O sistema tem **três peças**. As duas primeiras rodam sempre; a terceira só quando o equipamento muda.

| peça | pergunta que responde | quando roda | quem roda | já está no pacote? |
|---|---|---|---|---|
| **1. Inferência** (alerta de anomalia) | "tem algo anormal no equipamento agora?" | a cada lote de dado (ex.: de hora em hora ou diário) | integração, automático | sim (`simpred_inference.py`) |
| **2. Monitor de mudança** | "o normal do equipamento ainda é o que o modelo aprendeu?" | 1× por semana | integração, automático | sim (`monitor_drift.py`) |
| **3. Retreino** | "precisa de um modelo novo?" | só depois de um aviso do monitor **confirmado pela operação** | time de modelos | não: roda no repositório de modelos / ClearML e entrega um bundle novo |

```
 dado bruto (1 min) ──► 1. INFERÊNCIA ──► coluna `alerta` ──► operação verifica o equipamento
        │                      │
        │                      └── CSV da inferência ─┐
        └─────────────────────────────────────────────┴──► 2. MONITOR (semanal) ──► 🟢 / 🟡 / 🔴
                                                                                     │ 🟡 🔴
                                                                       time de modelos + operação
                                                                       "houve reparo? é normal novo?"
                                                                                     │ sim
                                                                     3. RETREINO (1, 3, 6 e 12 meses)
                                                                                     │
                                                         bundle novo ──► sombra ──► troca de pasta
```

Dois sinais diferentes, que **não se misturam**:

| | alerta do modelo (peça 1) | aviso de mudança (peça 2) |
|---|---|---|
| significa | anomalia nos sensores, que pode indicar problema no equipamento | o "normal" mudou desde o treino (reparo, regime de operação, sensor) |
| vai para | a operação, que verifica o equipamento | o time de modelos, que pergunta à operação |
| consequência | inspeção / manutenção | modelo novo, se for normal novo |

O monitor **nunca desliga nem silencia** o alerta. Trocar de modelo também não: o modelo novo continua alertando.

---

## 2. Peça 1 — inferência (o alerta)

Já documentada no `README.md` e no `MIGRACAO.md`. O essencial para a integração:

- **Entrada:** CSV bruto com 1 linha por minuto, 1ª coluna o timestamp, demais colunas os sensores (mesmo formato
  do exemplo em `B-8802B-2025/dados/`).
- **Chamada:** `prever(bundle, modelo, X, df_bruto=df)`. Passe sempre `df_bruto` com o mesmo CSV da entrada: é ele
  que deixa os minutos de bomba parada fora do alarme.
- **Saída:** por instante, `reconstruction_error`, `severity` (`normal` / `atencao` / `alarme`) e **`alerta`**.
  **O que vai para a operação é a coluna `alerta`**: o alarme que durou 1 hora ou mais.
- **Como o alerta é calculado:** o modelo (VAE) tenta reconstruir os 5 sensores; o erro acima de
  média + 6,5 desvios do erro normal conta como "acima do limiar". Vira alarme quando 15 dos últimos 20 pontos de
  5 min estão acima (sem contar parada), e vira alerta se o alarme durar 1 hora ou mais. Tudo isso está no
  `alarm.json` do bundle; nada fica no código.
- **Texto sugerido para a operação:** "anomalia real nos sensores do B-8802B (sensor que mais pesou: X), pode
  indicar problema no equipamento, favor verificar". Não chamar de falha.
- **Guarde o CSV da inferência**: o monitor semanal usa ele.

---

## 3. Peça 2 — monitor de mudança (semanal)

Detalhe completo em `MONITORAMENTO.md` (seção A). O essencial:

- **Roda 1× por semana**, depois da inferência, sobre o CSV dela e o CSV bruto do mesmo período. Não altera nada
  na inferência.
- **Saída:** status 🟢 / 🟡 / 🔴, o motivo (qual indicador e qual sensor), uma tabela por semana (`--csv`) e uma
  figura (`--png`).
- **O que a integração faz:** 🟢 arquiva; 🟡 avisa o time de modelos (mesma semana); 🔴 avisa no mesmo dia e passa a
  tratar os alertas do modelo com ressalva até a revisão. Junto do aviso, mandar as manutenções e intervenções do
  período, se houver.
- **Como detecta a mudança** (resumo):
  - **teste KS diário (M6):** compara a distribuição de cada sensor, dia a dia, com o período de treino do modelo
    (referência fixa, a janela de comparação desliza); avisa se 3 de 5 dias ficam diferentes;
  - **temperatura de mancal descontada a estação (M8):** compara a temperatura com a esperada pelo motor e pelas
    pressões (regressão), e avisa se 3 de 5 dias ficam fora da faixa normal;
  - **dado congelado (M7):** 3 ou mais sensores parados no mesmo valor por 12 h ou mais é falha de aquisição; sai
    das contas e gera aviso à parte (avisar também a instrumentação);
  - M1 a M5: taxa de alarme, erro relativo, cobertura e sensores fora da faixa do treino.
- **A referência dos detectores anda com o modelo:** cada bundle traz `drift_ref.json` (M6) e `residual_ref.json`
  (M8) calibrados no período de treino dele. Trocou o bundle, trocou a referência; não há nada a configurar.

---

## 4. Peça 3 — o retreino

### 4.1 Quando acontece

O retreino **não tem data marcada e nunca é automático**. Ele só acontece quando:

1. o monitor sai do verde, **e**
2. a operação confirma que a mudança é um **normal novo** (reparo, troca de peça, mudança de regime de operação).

Se a mudança for **degradação** do equipamento, **não se retreina**: o alerta é justamente o que interessa.
Se for **sensor trocado ou recalibrado**, só se recalibra (sem retreino).

> Por que não automático: testamos retreinar sozinho todo mês com os últimos 12 meses (janela deslizante de treino,
> `notebooks/drift/resultados_retreino_B-8802B_out2026.ipynb`, seção 5). Numa degradação lenta simulada, o modelo
> aprendeu o defeito como normal: o alerta caiu de ~10 % para ~2 % do mês. A confirmação da operação é a proteção.

### 4.2 O calendário: 1, 3, 6 e 12 meses

Depois da confirmação, o time de modelos treina modelos novos **só com dado depois da mudança**:

| modelo | quando fica pronto (a partir da data da mudança) | observação |
|---|---|---|
| 1 mês | com ≥ 1 mês **e** ≥ 300 h de bomba operando | já protege o equipamento; alertas com ressalva (marcado `provisional` no `alarm.json`) |
| 3 meses | 3 meses | |
| 6 meses | 6 meses | |
| 12 meses | 12 meses (≥ 4000 h) | cobre as estações do ano; **fica fixo** até a próxima mudança confirmada |

- A contagem começa **na data da mudança**, não na resposta da operação. Se a confirmação chegar depois de 1 mês, o
  modelo de 1 mês já pode ser treinado na hora.
- **No intervalo** (do aviso até o modelo de 1 mês), o modelo atual continua rodando; os alertas dele valem com
  ressalva, porque podem ser só o normal novo. A operação verifica do mesmo jeito.
- **Por que 1, 3, 6 e 12 e não todo mês:** no replay do B-8802B (jan/2025 a ago/2026), refazer todo mês deu 2
  alertas falsos (0,04 % do tempo) com 11 trocas de modelo; 1/3/6/12 deu 3 alertas (0,09 %) com 3 trocas. Só o de
  1 mês, sem refazer, deu 57 alertas: um modelo de 1 mês não conhece as estações.
- **Revisão anual:** com tudo verde, uma vez por ano o time de modelos repete os testes com o modelo em produção.
  Sem aviso confirmado, não se retreina.

### 4.3 O retreino usa AutoML?

**Não.** O AutoML (`scripts/automl.py`, busca ampla de arquiteturas no ClearML) foi usado **uma vez**, para escolher
a arquitetura do modelo de 2025 (VAE, camadas 128-64-32, latente 16, preset de pré-processamento `baseline`).

O retreino usa uma **grade fixa e pequena em volta dessa arquitetura** (`src/transpetro_modelos/drift/retrain.py`):

- 4 configurações de rede (camadas 128-64-32, 64-32-16 e 256-128-64; latente 16; taxa de aprendizado 1e-4 ou 1e-3)
  × 3 sementes = **12 candidatos**;
- até 60 épocas, parada antecipada com paciência 10;
- **escolha do candidato:** entre os que dão poucos alertas falsos na validação, o que pega mais cedo uma falha
  simulada; o escolhido ainda passa pelo teste de aceitação (4.5). Reprovou, tenta o próximo (até 3).

Por que não AutoML a cada retreino: leva muito mais tempo, o resultado varia de uma rodada para outra, e escolher
só pelo menor alerta falso tende a escolher o modelo mais "cego". A grade fixa é rápida, reprodutível e o teste de
aceitação garante a qualidade. **Rodar o AutoML de novo só se:** for um equipamento novo, mudarem os sensores, ou
nenhum candidato da grade passar no teste de aceitação.

### 4.4 Requisitos do retreino

**Dados**

| requisito | valor |
|---|---|
| frequência | 1 linha por minuto, mesmo formato do CSV de inferência, mesmos nomes de coluna |
| sensores do modelo | Pressão Sucção, Pressão Descarga, Vibração Bomba LA, Vibração Bomba LNA, Temperatura Bomba LA |
| sensores do monitor M8 | Temperatura Motor LA e LNA (preditores da temperatura do mancal) |
| volume mínimo | modelo de 1 mês: ≥ 1 mês e ≥ 300 h com a bomba operando; modelo de 12 meses: 12 meses e ≥ 4000 h |
| período | só depois da mudança confirmada; a operação confirma que o período é operação normal |
| limpeza | automática: bomba parada, partidas, manobras de pressão e dado congelado saem do treino. Paradas longas e falhas de comunicação devem ser informadas, não precisam ser removidas à mão |
| dado para o teste | o CSV da falha de 2022 do B-8802B (já no pacote, `B-8802B/dados/`) |

Onde o dado precisa estar, **hoje**: o retreino lê o dado do equipamento do Dataset do ClearML
(`transpetro-b-8802b-2025`, ou o arquivo local `Dados-novos/B-8802B-2025.feather`) e o teste de aceitação lê o CSV
em `B-8802B-2025/dados/`. Para um retreino novo, **a integração precisa disponibilizar o CSV bruto do período novo**
ao time de modelos, que atualiza o Dataset. (Ponto a melhorar: o pipeline aceitar o CSV direto.)

**Infraestrutura**

| requisito | valor |
|---|---|
| código | repositório de modelos (`src/transpetro_modelos` + `scripts/retrain_pipeline.py`), Python 3.12, dependências do `pyproject.toml` (via `uv`). **Não** faz parte do pacote autocontido de inferência |
| onde roda | no ClearML (fila `default`, imagem `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) ou em qualquer máquina com o repositório |
| hardware | **CPU basta**: a rede é pequena (5 sensores). GPU não é necessária |
| memória | ~1 GB de RAM |
| disco | bundle de saída ~216 KB; o dado de 19 meses tem ~845 mil linhas |

**Desempenho medido (B-8802B)**

| etapa | tempo |
|---|---|
| treinar 1 candidato com 12 meses de dado, CPU de 12 threads | ~45 s |
| retreino completo (12 candidatos + empacotar + teste de aceitação), worker do ClearML | 2 min (1 mês de dado) a 8 min (12 meses) |
| retreino completo em CPU comum | ~10 a 12 min com 12 meses |
| inferência sobre 19 meses de dado (845 mil linhas) | < 1 s de modelo + ~5 s para ler o CSV |
| monitor semanal sobre 19 meses | ~12 s, < 1 GB de RAM |

**Pessoas**

| quem | o quê | prazo sugerido |
|---|---|---|
| integração | avisar o time de modelos quando o monitor sai do verde; disponibilizar o CSV bruto do período | 🟡 mesma semana, 🔴 mesmo dia |
| operação | dizer o que aconteceu (reparo, troca, regime) e se o período novo é operação normal | até 2 semanas |
| time de modelos | investigar, retreinar, validar e entregar o bundle | até 1 semana depois de ter o dado mínimo |

### 4.5 Teste de aceitação (o que o modelo novo precisa provar)

Nenhum modelo novo é entregue sem passar. Os critérios ficam em `src/transpetro_modelos/drift/battery.py`.

| teste | modelos de 1, 3 e 6 meses | modelo de 12 meses (revisão com held-out) |
|---|---|---|
| alerta falso no período normal novo | ≤ 0,5 % do tempo na validação (fim da janela de treino) | ≤ 0,05 % no held-out (meses depois do treino) e ≤ 0,30 % no período posterior |
| falha real de 2022 do B-8802B | alarme ≥ 0,5 dia antes da quebra (o alarme no normal de 2022 é só informativo: é o normal de antes do reparo) | alarme ≥ 2 dias antes, e ≤ 0,5 % de alarme no normal de 2022 |
| falha simulada (assinatura medida da falha de 2022, injetada no período novo) | detectada antes do fim da rampa de 48 h | detectada ≥ 20 h antes, e também em meia intensidade |
| episódios reais que devem continuar alertando | — | o alerta de 17/01/2026 (vibração LA + LNA) |

### 4.6 O que a integração recebe e o que faz

1. **Um bundle novo em pasta nova**, ao lado do atual (ex.: `B-8802B-2026/modelos/model_<início>_<fim>_VAE/`), com
   os mesmos arquivos de sempre (`model_state.pt`, `model_arch.json`, `scaler.pkl`, `clip_bounds.json`,
   `pipeline.json`, `alarm.json`) **mais** `drift_ref.json` e `residual_ref.json`. O código de inferência e do
   monitor **não muda**: só o caminho do bundle.
2. **Sombra:** rodar o bundle atual e o novo em paralelo (inferência e monitor), sem trocar nada:
   - modelos de 1, 3 e 6 meses: 1 a 2 semanas;
   - modelo de 12 meses: 4 semanas.
3. **Troca**, com o aval do time de modelos: o novo está verde no monitor e as diferenças de alerta entre os dois
   foram entendidas. Troca = apontar o script para a pasta nova.
4. **Rollback:** a pasta antiga fica guardada; voltar = apontar de novo para ela.

Para a integração, isso significa receber **até 4 bundles** depois de cada mudança confirmada (1, 3, 6 e 12 meses),
e depois nenhum até a próxima mudança.

---

## 5. Desempenho do sistema (B-8802B)

Medido rodando a política inteira sobre o dado real de jan/2025 a ago/2026, como se estivesse em produção
(replay no ClearML; `notebooks/drift/resultados_retreino_B-8802B_out2026.ipynb`).

| métrica | valor |
|---|---|
| alertas falsos, modelo de 2022 mantido | 269 alertas, 12 % do tempo em alerta |
| alertas falsos, com a política (depois do 1º modelo novo) | **3 alertas, 0,09 % do tempo** (19/05/2025, 10/01/2026, 17/01/2026) |
| aviso de mudança depois do reparo | 09/01/2025 → modelo de 1 mês em 06/02/2025 |
| aviso de mudança de 2026 (mancal LA ~6 °C mais frio) | 01/04/2026, 4 dias depois da parada de 27/03 |
| falha real de 2022 (nenhum modelo novo a viu no treino) | os 4 modelos alertam em 04/07 de manhã, ~2,1 dias antes da quebra, e o alerta fica ligado até ela |
| alerta falso no normal de 2022 | modelo de 12 meses: 0; modelos de 1, 3 e 6 meses: 5, 1 e 1 |

**Limites, para não prometer o que o sistema não faz:**

- O modelo **detecta a anomalia quando ela aparece nos sensores**; não prevê semanas antes. Na falha de 2022 a
  vibração ficou estável até 04/07 e a bomba quebrou em 06/07; uma regra simples de limite na vibração dispararia
  praticamente junto. O ganho do modelo sobre um limite fixo ainda não foi medido.
- No mês entre o aviso e o modelo de 1 mês, o modelo antigo ainda pode alertar à toa.
- Os modelos de 1 a 6 meses são menos maduros (mais alerta falso fora do regime em que treinaram).
- Degradação muito lenta e sutil pode não ser detectada (caso do B-4703).

---

## 6. Checklist de integração

1. ☐ Instalar o pacote (`requirements.txt`, Python 3.12) e rodar o exemplo do B-8802B-2025.
2. ☐ Integrar a inferência com `df_bruto` e encaminhar a coluna `alerta` à operação, com o texto sugerido (seção 2).
3. ☐ Guardar o CSV de cada inferência e o CSV bruto do mesmo período.
4. ☐ Agendar o monitor 1× por semana (seção 3); guardar saída de texto, `--csv` e `--png`.
5. ☐ Criar o canal de aviso: 🟡 / 🔴 → time de modelos, com o motivo e as manutenções do período.
6. ☐ Combinar com a operação quem responde "o que foi feito?" e em quanto tempo.
7. ☐ Preparar a troca por pasta: o caminho do bundle em configuração, não no código, para sombra e rollback.
8. ☐ Combinar como tratar os alertas no intervalo entre o aviso e o modelo novo (verificar, com ressalva).

---

## 7. Perguntas frequentes

**O retreino é automático?** Não. Só com o aviso do monitor e a confirmação da operação (seção 4.1).

**Usa AutoML?** Não; uma grade fixa de 12 candidatos em volta da arquitetura escolhida pelo AutoML (seção 4.3).

**Precisa de GPU?** Não. Em CPU comum o retreino completo leva ~10 a 12 min.

**Quanto tempo do aviso até o modelo novo?** No mínimo 1 mês depois da mudança (dado mínimo), mais a resposta da
operação e ~1 semana do time de modelos. No B-8802B: aviso em 09/01/2025, modelo de 1 mês em 06/02/2025.

**E se a operação não responder?** Nada é retreinado. O modelo atual continua; o monitor continua avisando.
Hoje é o caso do B-8802B: aviso em 01/04/2026, aguardando o que foi feito nas paradas de 17/03 e 27/03.

**O monitor pode rodar todo dia?** Pode, mas os indicadores foram calibrados por semana (ex.: 3 de 5 dias, 2 de 4
semanas); rodar mais vezes não muda o resultado, só repete.

**O que muda no código quando chega um bundle novo?** Nada; só o caminho da pasta do bundle.

**Um bundle sem `drift_ref.json` ou `residual_ref.json`?** A inferência funciona; o monitor pula o indicador
correspondente (M6 ou M8) e avisa.

**Por que usar só dado depois da mudança, e não os últimos 12 meses?** O dado de antes é o normal antigo. Misturar
ensina o modelo a aceitar os dois, e ele fica menos sensível. Numa mudança pequena (2026) misturar funcionou
bem; numa mudança grande ainda não testamos (em aberto).

---

## 8. Em aberto

1. O pipeline de retreino aceitar o CSV bruto direto (hoje lê do Dataset do ClearML).
2. Comparar o modelo com uma regra simples de limite em 2025–26 (quantos alertas falsos cada um daria).
3. Testar "últimos 12 meses" × "só dado novo" numa mudança grande (B-4064A, +20 °C depois do reparo).
4. Resposta da operação sobre as paradas de 17/03 e 27/03/2026 do B-8802B.
