# Visão Geral do Projeto — Detecção de Anomalias (PdM Transpetro)

> Documento de overview para entendimento do projeto e planejamento do **deploy**.
> Atualizado em 08/jun/2026. Branch de referência: `main` (já contém a integração da branch `Lara`).

---

## 1. O que o projeto faz

Manutenção preditiva (PdM) em bombas/turbocompressores da Transpetro por **detecção de
anomalias não-supervisionada**. A ideia central:

1. O modelo aprende o padrão de **operação normal** de cada equipamento (a partir dos dados
   anteriores à falha).
2. Em operação, ele recebe a telemetria e calcula um **score de anomalia** (erro de
   reconstrução, para autoencoders).
3. Quando o score ultrapassa um **limiar** por tempo suficiente (debounce), dispara um **alarme**.

**North star:** detectar a falha o mais cedo possível **com o mínimo de falsos positivos**
(um alarme que dispara o tempo todo é inútil em operação).

Cada equipamento tem **uma falha histórica conhecida** usada para validar o detector. Por isso
existe também uma camada de validação por **falhas sintéticas** (injeção de rampas) para não
depender de um único evento real.

---

## 2. Modelos suportados

O runner de AutoML varre vários tipos de modelo e hiperparâmetros por equipamento:

| Modelo | Tipo | Quando costuma vencer |
|---|---|---|
| **Dense Autoencoder** | reconstrução (PyTorch) | caso geral; venceu em B-8802B e B-6511502A |
| **LSTM Autoencoder** | reconstrução temporal (PyTorch) | sinais com dinâmica temporal |
| **VAE** | reconstrução probabilística (PyTorch) | alternativa ao dense |
| **OCSVM** | fronteira (scikit-learn) | datasets pequenos |
| **Isolation Forest** | ensemble de árvores (scikit-learn) | rápido, baseline |
| **LOF** | densidade local (scikit-learn) | anomalias locais |

---

## 3. Como o repositório está organizado

```
Transpetro-modelos/
├── src/transpetro_modelos/        ← biblioteca (lógica reutilizável)
│   ├── config.py                  ← ★ catálogo de equipamentos (EquipmentConfig) + presets
│   ├── data/
│   │   ├── loading.py             ← carrega feather/csv local OU via ClearML Dataset
│   │   ├── preprocessing.py       ← pipeline configurável (filtros, resample, normalize, resíduo…)
│   │   │                            + PreprocessingArtifacts (scaler/clip/coefs ajustados)
│   │   ├── splitting.py           ← split temporal train/val/test (val_start/val_end)
│   │   └── upload_datasets.py     ← envia os dados locais para o ClearML
│   ├── models/
│   │   └── autoencoder.py         ← DenseAutoencoder / LSTMAutoencoder (PyTorch)
│   └── training/
│       ├── automl.py              ← ★ grid search, run_trial, ranking, seleção
│       ├── train.py               ← loop de treino (early stopping, scheduler)
│       └── evaluate.py            ← erro de reconstrução, threshold, métricas, debounce, CUSUM
│
├── scripts/                       ← entry points (linha de comando)
│   ├── automl.py                  ← ★ runner principal de AutoML (é o que usamos)
│   ├── train_equipment.py         ← treino simples de 1 equipamento
│   ├── train_all.py               ← treina todos
│   └── upload_data.py             ← upload dos datasets (rodar uma vez)
│
├── notebooks/                     ← análise de resultados por equipamento
│   └── README.md                  ← ★ aponta o MELHOR notebook/resultado de cada equipamento
│
├── docs/
│   ├── OVERVIEW.md                ← este documento
│   ├── auditoria_pdm.md           ← auditoria metodológica + ajustes (held-out, debounce, CUSUM)
│   ├── analise_b4064a_regime.md   ← caso do regime térmico pós-reparo do B-4064A
│   └── analise_b0302c.md          ← por que o B-0302C é fraco (triagem det×FP)
│
├── Dados/ Dados-novos/ DadosV2/   ← dados brutos locais (feather/csv)
├── pyproject.toml / uv.lock       ← dependências (gerenciadas por uv)
└── README.md                      ← guia rápido de uso
```

O `★` marca os arquivos centrais. **`config.py` é o coração**: tudo o que varia entre
equipamentos (sensores, filtros, janelas, presets de preprocessing) está declarado lá.

---

## 4. O fluxo de ponta a ponta

```
   Dados brutos              ClearML Dataset            AutoML (worker GPU)
  (feather/csv)   ──upload──▶  transpetro-<eq>   ──────▶  scripts/automl.py
                                                              │
                                                              ▼
                         preprocessing (por equipamento, config.py)
                                                              │
                                                              ▼
                              split temporal  train / val / test
                                                              │
                                                              ▼
                    grid search: N modelos × hiperparâmetros × presets
                                                              │
                                                              ▼
                     seleção do melhor trial (det. máx. sujeita a FP ≤ limite)
                                                              │
                                                              ▼
                  artifacts na task: modelo + scores + trial vencedor
                                                              │
                                                              ▼
                  notebook de resultado (re-threshold, validação, plots)
```

### Configuração por equipamento (`EquipmentConfig`)

Cada equipamento define:

- **`failure_date`** + descrição da falha
- **`dataset_name`** (no ClearML) e/ou **`local_feather`** (caminho local)
- **`pre_split_steps`**: limpeza aplicada ANTES do split (resample, filtros, remoção de
  transientes) — usa o dado bruto inteiro
- **`preprocessing_steps`** / **`preprocess_presets`**: transformações ajustadas só no treino
  (clip, normalize, resíduo de carga…) e reaplicadas em val/test sem vazamento
- **`val_start_date` / `val_end_date`**: janela de validação fixa (held-out)
- **`prefailure_days` / `normal_end_days`**: janelas de avaliação (onde se mede detecção e FP)

### Seleção do modelo (importante para confiar no resultado)

A seleção padrão maximiza a **detecção pré-falha** sujeita a uma **restrição dura de falsos
positivos** (`--max-fp-rate`), medindo o FP de forma honesta na validação held-out
(`--select-by heldout`). Detalhes e o histórico da auditoria estão em
[`docs/auditoria_pdm.md`](auditoria_pdm.md).

---

## 5. Como rodar (treinar)

Pré-requisitos: [`uv`](https://docs.astral.sh/uv/) e ClearML configurado (`clearml-init`).

```bash
uv sync                                   # cria .venv e instala dependências
uv run python scripts/upload_data.py      # 1x: sobe os datasets pro ClearML
```

Treino via AutoML (recomendado), remoto no worker GPU:

```bash
uv run python scripts/automl.py \
  --equipment B-8802B \
  --remote --mode full \
  --max-fp-rate 0.01 \
  --select-by heldout \
  --clearml-task-name "automl-b8802b-prod"
```

- `--mode quick|full|extensive` controla o tamanho do grid.
- `--remote` submete pra fila do ClearML e encerra a execução local (pode fechar o terminal).
- **Nunca usar `--local-data` em execução remota** (o worker não enxerga o disco local).

Equipamentos disponíveis (14): ver `--help` ou a lista em `config.py`.

---

## 6. Status atual da frota

| Equipamento | Status | Resultado |
|---|---|---|
| **B-8802B** | ✅ bom | dense, 55,6% detecção @ 0,00% FP |
| **B-6511502A** | ✅ bom | dense, 68,9% @ 0,06% FP, lead ~49h |
| **B-4064A** | ✅ bom | detector 71% @ 0% + config de produção re-baselinada para 2025-26 |
| **B-0302C** | ⚠️ fraco | 30% @ 0,87% — sensores de motor zerados (documentado) |
| **B-4703.24001B** | ⚠️ fraco | 0,8% — desgaste de rolamento exige espectro (documentado) |
| **B-402E / B-5401A** | ❌ descartados | instrumentação insuficiente para o modo de falha |
| **B-3403C / B-90001A (interpolados)** | ↪ branch Lara | configs integrados; resultados nos notebooks |

O notebook recomendado de cada equipamento está em [`notebooks/README.md`](../notebooks/README.md).

---

## 7. Deploy — como colocar em produção

> Esta seção é o roteiro para o deploy. **Importante:** o projeto hoje é uma pipeline de
> **treino/validação** (ClearML). Falta a etapa de **empacotar e servir** o modelo. Abaixo o
> que já existe, o que falta e os passos.

### 7.1 O que o treino produz hoje (artifacts da task ClearML)

Ao final de um `automl.py`, a task guarda:

| Artifact | Conteúdo |
|---|---|
| `best_model.pt` (ou `.pkl`) | pesos do modelo vencedor (state_dict PyTorch; pickle para OCSVM/IF/LOF) |
| `best_trial.pkl` | o `TrialConfig` vencedor + métricas (inclui o **threshold** e o **preset**) |
| `best_full_scores.parquet` | score de anomalia por timestamp na série completa |
| `preprocessing.pkl` | **`PreprocessingArtifacts`** ajustado no treino (scaler, clip_bounds, coefs do resíduo) |
| `automl_results` | tabela com todos os trials (para auditoria/comparação) |

### 7.2 Por que o `preprocessing.pkl` é essencial (resolvido)

Para pontuar **dados novos** em produção é preciso reaplicar exatamente o mesmo preprocessing
do treino. A normalização aprende a média/desvio de cada sensor no treino (`scaler.fit`) e o
modelo é treinado esperando os dados nessa escala. Se em produção o scaler fosse reajustado na
janela nova (`fit` em vez de `transform`), a escala mudaria → o erro de reconstrução perderia
o sentido → o threshold calibrado deixaria de valer.

Por isso o treino agora **persiste o `PreprocessingArtifacts`** (scaler, clip_bounds e
coeficientes do resíduo de carga) como `preprocessing.pkl`, salvo localmente e enviado como
artifact da task ClearML. Em produção, basta recarregá-lo e passar como `fitted_artifacts`
para o `run_preprocessing` — o pipeline reaplica a MESMA escala do treino, sem vazamento.

> Modelos treinados **antes** desta mudança não têm o `preprocessing.pkl`. Para colocá-los em
> produção, basta re-treinar uma vez (o artifact passa a sair automaticamente) ou extrair o
> scaler do notebook de análise correspondente, que o recalcula.

### 7.3 O conceito: um "bundle" por equipamento

A interface de inferência é **única**; o que muda entre equipamentos é o **conteúdo** do bundle
(absorvido pelo `EquipmentConfig`). Cada bundle de produção contém:

```
bundle_B-8802B_v1/
├── model.pt                  # pesos do modelo
├── preprocessing.pkl         # PreprocessingArtifacts (scaler, clip_bounds, coefs)  ← já gerado pelo treino
├── alarm.json                # threshold, debounce (k/n), política de alarme
├── equipment_config.json     # snapshot do EquipmentConfig (sensores, steps, presets)
└── metadata.json             # equipamento, data de treino, task_id ClearML, métricas
```

### 7.4 Passo a passo recomendado para o deploy

**Passo 1 — Persistir o preprocessing no treino. ✅ FEITO.**
`scripts/automl.py` (`_save_best_artifacts` / `_retrain_best`) já salva o
`PreprocessingArtifacts` do trial vencedor como `preprocessing.pkl` e o sobe como artifact da
task. (Modelos treinados antes desta mudança precisam ser re-treinados uma vez — ver §7.2.)

**Passo 2 — Script de exportação `scripts/export_model.py`.**
Dado um `task_id` do ClearML + os ajustes finais do notebook (percentil/debounce escolhidos
na validação), monta o diretório `bundle_<eq>_v<n>/` da seção 7.3. Versionado por equipamento.

**Passo 3 — Módulo de inferência `predict.py` (interface única).**
```python
bundle = load_bundle("bundle_B-8802B_v1/")
result = bundle.predict(df_raw)   # aplica config → preprocessing → modelo → score → alarme
# result: score por timestamp + flag is_anomaly (já com debounce)
```
A mesma função serve todos os equipamentos; as diferenças vivem no config dentro do bundle.

**Passo 4 — Definir o alvo de execução (decisão com a equipe).**
Como os alarmes serão consumidos? Opções:
- **Batch agendado** (ex.: roda a cada hora sobre a janela recente) — mais simples, recomendado para começar;
- **Serviço/API** (recebe telemetria, devolve score) — para integração online;
- **Edge** (junto ao SCADA/historiador) — menor latência, mais complexo.

Isso define se o bundle vira pickle num cron, um container com FastAPI, ou um job no ClearML.

**Passo 5 — Origem dos dados em produção.**
Definir de onde vem a telemetria nova (PI/historiador, banco, arquivo) e em que cadência.
O `loading.py` já abstrai a leitura; basta um adaptador para a fonte de produção.

**Passo 6 — Monitoramento e re-treino.**
- Registrar taxa de alarmes em produção (detectar drift / aumento de FP).
- Re-baselinar após reparos/troca de regime (o caso do **B-4064A** mostra que mudança de
  regime térmico pós-reparo exige re-treino — ver [`docs/analise_b4064a_regime.md`](analise_b4064a_regime.md)).

### 7.5 Ordem sugerida

1. Passo 1 (persistir preprocessing) — ✅ já concluído (era o pré-requisito que bloqueava o resto).
2. Passos 2–3 (export + predict) — produzem o artefato deployável e a interface.
3. Passo 4 (definir alvo) — **decisão de equipe**, não de código.
4. Passos 5–6 — integração com a fonte real e operação.

---

## 8. Limitações conhecidas (para alinhar expectativas)

- **Sem análise espectral:** a cadência máxima dos dados (~60s) não captura bandas de
  frequência. Falhas de **banda estreita** (desgaste de rolamento BPFO/BPFI/BSF,
  desbalanceamento, desalinhamento) são **indetectáveis precocemente** — só se pega o que
  eleva a energia global de vibração. Explica B-4703 e B-0302C.
- **Uma falha real por equipamento:** a validação é reforçada por falhas sintéticas, mas
  nenhuma métrica substitui mais eventos reais.
- **Cobertura de sensores varia:** alguns equipamentos têm canais zerados/saturados que
  limitam o que é detectável (documentado por equipamento nos `docs/`).
