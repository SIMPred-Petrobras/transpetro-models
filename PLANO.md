# Plano: Autoencoders para Detecção de Anomalias - Transpetro

## Contexto

Projeto de manutenção preditiva para bombas da Transpetro. Temos 4 equipamentos com dados de sensores (pressão, vibração, temperatura, corrente) e datas de falha conhecidas. O autoencoder aprende o padrão de operação normal e detecta anomalias via erro de reconstrução elevado. Usaremos **uv** para gerenciamento de pacotes e **ClearML** para execução remota e tracking de experimentos.

## Dados Existentes

| Equipamento | Linhas | Features | Falha | Data Detecção |
|-------------|--------|----------|-------|---------------|
| B-402E | 1.2M | 15 + datetime col | Quebra barra rotor + colisão estator | 2019-10-30 |
| B-4064A | 178K | 9 (sem datetime col) | Roçamento interno rotor | 2024-08-30 |
| B-8802B | 98K | 8 (sem datetime col) | Trinca lâminas acoplamento | 2022-07-06 |
| B-90001A | 395K | 9 (sem datetime col) | Afrouxamento parafusos mancal | 2021-08-28 |

Todos float64, sem nulls. `falhas.xlsx` tem metadados das falhas.

---

## Estrutura do Projeto

```
Transpetro-modelos/
├── pyproject.toml
├── .python-version          # 3.12
├── .gitignore
├── Dados/                   # dados locais (gitignored)
├── src/
│   └── transpetro_modelos/
│       ├── __init__.py
│       ├── config.py            # metadata equipamentos, datas falha
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loading.py       # carrega feather (local ou ClearML)
│       │   ├── preprocessing.py # filtra on/off, remove transientes, StandardScaler
│       │   ├── splitting.py     # split temporal train/val/test
│       │   └── upload_datasets.py  # upload único para ClearML Dataset
│       ├── models/
│       │   ├── __init__.py
│       │   └── autoencoder.py   # PyTorch dense autoencoder
│       └── training/
│           ├── __init__.py
│           ├── train.py         # loop de treino com early stopping
│           └── evaluate.py      # erro reconstrução, threshold, métricas
└── scripts/
    ├── upload_data.py       # entry point upload datasets
    ├── train_equipment.py   # entry point treino (1 equipamento, ClearML Task)
    └── train_all.py         # lança treino para todos os 4 equipamentos
```

## Passo a Passo de Implementação

### Passo 1: Inicializar projeto com uv

- `uv init` no diretório do projeto
- Configurar `pyproject.toml` com dependências:
  - `torch>=2.2.0`, `pandas>=2.2.0`, `pyarrow>=15.0.0`
  - `scikit-learn>=1.4.0`, `matplotlib>=3.8.0`
  - `clearml>=1.16.0`, `openpyxl>=3.1.0`
- `uv sync` para criar `.venv` e instalar tudo
- Criar `.gitignore` (Dados/, .venv/, __pycache__/, *.pyc)

### Passo 2: Configuração dos equipamentos (`src/transpetro_modelos/config.py`)

Dataclass `EquipmentConfig` com:
- `equipment_id`, `failure_date`, `failure_description`
- `datetime_column` (só B-402E tem como coluna, os outros têm como índice)
- `on_off_column` + `on_off_threshold` (para filtrar períodos desligados)
- `exclusion_days_before` (dias antes da falha excluídos do treino, default=10)
- `dataset_name` para ClearML (ex: `"transpetro-b-402e"`)
- `preprocessing_steps`: lista configurável de etapas de preprocessing (ver Passo 4)

### Passo 3: Upload dos datasets para ClearML (OBRIGATÓRIO para execução remota)

**Sim, é necessário subir os datasets.** O ClearML Agent remoto não tem acesso aos arquivos locais. Usaremos `clearml.Dataset`:

```python
# scripts/upload_data.py
from clearml import Dataset

def upload(equipment_id, file_path):
    ds = Dataset.create(
        dataset_name=f"transpetro-{equipment_id.lower()}",
        dataset_project="Transpetro",
    )
    ds.add_files(file_path)
    ds.upload()
    ds.finalize()
```

- Upload cada .feather como dataset separado
- Upload `falhas.xlsx` como `"transpetro-metadata"`
- Rodar uma vez localmente: `uv run python scripts/upload_data.py`

### Passo 4: Data loading e preprocessing (CONFIGURÁVEL POR EQUIPAMENTO)

**`data/loading.py`**:
- `load_equipment_data(equipment_id, from_clearml=True)` → DataFrame com DatetimeIndex
- Se `from_clearml`: usa `Dataset.get(dataset_name=...).get_local_copy()`
- Se local: lê de `Dados/`

**`data/preprocessing.py`** — Cada etapa é uma função independente. O pipeline de preprocessing é configurável via `preprocessing_steps` no config de cada equipamento:

```python
# Cada step é um dict com nome da função e seus parâmetros
# Exemplo config para B-402E:
preprocessing_steps = [
    {"step": "filter_running", "column": "Corrente", "threshold": 1.0},
    {"step": "remove_transients", "minutes": 10},
    {"step": "normalize", "method": "standard"},  # ou "minmax", "robust"
]

# Exemplo config para B-90001A (sem coluna de corrente, sem filtro on/off):
preprocessing_steps = [
    {"step": "normalize", "method": "standard"},
]
```

Funções disponíveis:
- `filter_running(df, column, threshold)` → remove linhas com bomba desligada
- `remove_transients(df, minutes=10)` → remove primeiros N min após religamento
- `remove_sensor_errors(df, error_values=[-25.0])` → troca códigos de falha por `NaN`
- `normalize(df, method="standard", scaler=None)` → StandardScaler/MinMaxScaler/RobustScaler
- `select_features(df, features)` → seleciona subset de colunas (caso queira treinar com menos features)
- `resample(df, freq="5min")` → reamostra dados para frequência diferente
- `ffill(df, limit=4)` → preenche faltantes para frente até `limit` períodos; depois remove linhas ainda com `NaN`
- `clip(df, lower_pct=1, upper_pct=99)` → limita extremos por percentis calculados no treino

A função `run_preprocessing(df, steps, fitted_scalers=None)` executa o pipeline na ordem definida. Isso permite:
- Treinar B-402E com filtro de corrente + remoção de transientes
- Treinar B-90001A só com normalização
- Experimentar diferentes combinações via hyperparâmetros no ClearML

Detalhe importante do `ffill` no caso `B-4064A-novos`:
- Config atual: `{"step": "resample", "freq": "1h"}` + `{"step": "ffill", "limit": 6}`
- Interpretação: preenche no máximo 6 horas consecutivas de gap
- Se o gap for maior que 6 horas, o excedente fica `NaN` e cai no `dropna()`

**`data/splitting.py`**:
- `temporal_split(df, failure_date, exclusion_days=10, val_fraction=0.2)`
- **Train**: início até `failure_date - exclusion_days`, primeiros 80%
- **Val**: mesma faixa, últimos 20% (para early stopping)
- **Test**: de `failure_date - exclusion_days` em diante (inclui degradação e falha)

### Passo 5: Modelo Autoencoder (`models/autoencoder.py`)

Dense autoencoder em PyTorch com dimensão de entrada configurável:

```
Encoder: Input(n) → Linear(64) → ReLU → BN → Linear(32) → ReLU → BN → Linear(16)
Decoder: Linear(16) → ReLU → BN → Linear(32) → ReLU → BN → Linear(64) → ReLU → Linear(n)
```

- Camadas do encoder escalam com `n_features`: 8 features → [32, 16, 8], 15 features → [64, 32, 16]
- Loss: MSE entre input e output reconstruído
- Sem ativação na camada final do decoder

### Passo 6: Script de treino com ClearML (`scripts/train_equipment.py`)

```python
from clearml import Task

task = Task.init(project_name="Transpetro", task_name=f"autoencoder-{equipment_id}")

hparams = {
    "equipment_id": "B-402E",
    "encoding_layers": [64, 32, 16],
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 100,
    "patience": 10,
    "exclusion_days": 10,
    "threshold_percentile": 95,
    "weight_decay": 1e-5,
}
task.connect(hparams)

# Para execução remota automática (descomente quando pronto):
# task.execute_remotely(queue_name="default")
```

Pipeline no script:
1. Carregar dados via ClearML Dataset
2. Preprocessar conforme `preprocessing_steps` do config do equipamento
3. Split temporal
4. Criar DataLoaders
5. Treinar com early stopping, logando `train_loss` e `val_loss` por epoch no ClearML (scalars)
6. Calcular erro de reconstrução no test set
7. Definir threshold (percentil 95 do erro no treino)
8. Salvar artifacts no ClearML (ver Passo 7)

### Passo 7: Artifacts e Coleta de Resultados

**O que é salvo como artifact na task do ClearML (não precisa de plots para rodar remotamente):**

```python
# Salvar modelo treinado (.pt)
task.upload_artifact("model", artifact_object=model.state_dict())
# Ou salvar como arquivo .pt:
torch.save(model.state_dict(), "model.pt")
task.upload_artifact("model_file", artifact_object="model.pt")

# Salvar scaler (para reproduzir preprocessing)
task.upload_artifact("scaler", artifact_object=scaler)

# Salvar threshold e métricas
task.upload_artifact("results", artifact_object={
    "threshold": threshold,
    "train_mse_mean": float(train_errors.mean()),
    "train_mse_std": float(train_errors.std()),
    "test_mse_mean": float(test_errors.mean()),
    "n_anomalies": int(anomalies.sum()),
})

# Salvar DataFrame de scores do test set (erro por timestamp)
task.upload_artifact("test_scores", artifact_object=scores_df)
```

**Como coletar os resultados depois da execução remota:**

```python
from clearml import Task

# 1. Buscar task pelo nome ou ID
task = Task.get_task(project_name="Transpetro", task_name="autoencoder-B-402E")

# 2. Baixar modelo treinado
model_path = task.artifacts["model_file"].get_local_copy()

# 3. Baixar métricas/threshold
results = task.artifacts["results"].get()  # retorna o dict direto

# 4. Baixar DataFrame com scores do test set
scores_df = task.artifacts["test_scores"].get()  # retorna pd.DataFrame

# 5. Ver scalars logados (loss curves)
scalars = task.get_reported_scalars()

# 6. Comparar tasks no dashboard ClearML:
#    - Abrir projeto "Transpetro" no browser
#    - Selecionar múltiplas tasks → Compare → ver métricas lado a lado
```

**Também logamos scalars (métricas numéricas) que ficam visíveis no dashboard:**
- `train_loss` e `val_loss` por epoch (curvas de aprendizado)
- `test_mse_mean`, `threshold`, `n_anomalies` como scalars finais

Plots são **opcionais** — podem ser gerados localmente depois com os artifacts baixados. Não são necessários para a execução remota.

### Passo 8: Avaliação local (opcional, pós-coleta)

**`training/evaluate.py`** — funções para análise local após baixar artifacts:
- `compute_reconstruction_errors(model, dataloader)` → MSE por amostra
- `determine_threshold(train_errors, percentile=95)` → float
- `score_test_set(model, test_df, scaler, threshold)` → DataFrame com `reconstruction_error` + `is_anomaly`
- Plots opcionais para análise visual (rodar localmente, não no ClearML)

### Passo 9: Execução remota no ClearML

**Duas opções:**

1. **Via código**: Adicionar `task.execute_remotely(queue_name="default")` após `Task.init()` — a execução local para, a task é enfileirada, o ClearML Agent executa remotamente
2. **Via UI**: Rodar localmente primeiro (cria a task), depois clonar no dashboard ClearML e enfileirar para um worker

O ClearML Agent instala dependências automaticamente a partir do que foi logado. Para GPU, setar `task.set_base_docker("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")`.

**Flexibilidade**: Para experimentar com preprocessing diferente por equipamento, basta clonar a task no ClearML UI, alterar os hyperparâmetros (`preprocessing_steps`, `encoding_layers`, etc.) e enfileirar novamente. Cada execução fica registrada com seus parâmetros e resultados.

---

## Ordem de Execução

1. `uv init` + configurar `pyproject.toml` + `uv sync`
2. Criar estrutura de diretórios e `__init__.py`
3. Implementar `config.py` com `preprocessing_steps` por equipamento
4. Implementar `data/loading.py` + `data/preprocessing.py` (pipeline configurável) + `data/splitting.py`
5. **Rodar `scripts/upload_data.py`** para subir datasets ao ClearML
6. Implementar `models/autoencoder.py`
7. Implementar `training/train.py` + `training/evaluate.py`
8. Implementar `scripts/train_equipment.py`
9. Testar localmente com B-8802B (menor dataset, iteração rápida)
10. Testar execução remota via ClearML
11. Rodar `scripts/train_all.py` para os 4 equipamentos
12. Coletar resultados: `task.artifacts["model_file"].get_local_copy()` + `task.artifacts["results"].get()`

## Verificação

- **Local**: `uv run python scripts/train_equipment.py --equipment B-8802B` completa sem erros
- **ClearML Dashboard**: Task aparece com scalars (loss curves) e artifacts (model .pt, scaler, results dict, test_scores DataFrame)
- **Coleta**: Conseguir baixar modelo e métricas via `Task.get_task()` + `.artifacts`
- **Remoto**: Clonar task no ClearML UI, enfileirar, verificar que executa e produz artifacts
- **Flexibilidade**: Clonar task, mudar `preprocessing_steps` nos hyperparâmetros, re-executar com pipeline diferente
