# Transpetro — Detecção de Anomalias com Autoencoders

Detecção de anomalias em bombas da Transpetro usando autoencoders densos (PyTorch). O modelo aprende o padrão de operação normal e sinaliza anomalias quando o erro de reconstrução supera um threshold.

**Equipamentos cobertos:**

| Equipamento | Falha | Data |
|-------------|-------|------|
| B-402E | Quebra de barra do rotor + colisão no estator | 2019-10-30 |
| B-4064A | Roçamento interno do rotor | 2024-08-30 |
| B-8802B | Trinca nas lâminas do acoplamento | 2022-07-06 |
| B-90001A | Afrouxamento dos parafusos do mancal | 2021-08-28 |

---

## Pré-requisitos

- [`uv`](https://docs.astral.sh/uv/) instalado
- ClearML configurado (`clearml-init` ou `clearml.conf` presente)

```bash
uv sync          # cria .venv e instala dependências
```

---

## Uso

### 1. Upload dos datasets para o ClearML (rodar uma vez)

Os datasets precisam estar no ClearML para que os workers remotos consigam acessá-los.

```bash
uv run python scripts/upload_data.py
```

Isso cria um `clearml.Dataset` para cada equipamento (`transpetro-b-402e`, etc.) e um `transpetro-metadata` com o `falhas.xlsx`. Só precisa rodar uma vez — ou novamente se os dados mudarem.

---

### 2. Treinar um equipamento

#### Localmente

```bash
uv run python scripts/train_equipment.py --equipment B-8802B
```

A task fica registrada no ClearML com métricas (loss curves) e artifacts (modelo, scaler, scores).

#### No servidor ClearML (desligar o notebook após submeter)

```bash
uv run python scripts/train_equipment.py --equipment B-402E --remote
```

Com `--remote`, o script registra a task, envia para a fila `default` do ClearML e **para a execução local**. O worker `cica:gpu0` executa tudo no servidor. Você pode fechar o terminal.

Equipamentos disponíveis: `B-402E`, `B-4064A`, `B-8802B`, `B-90001A`

---

### 3. Treinar todos os equipamentos

```bash
# Local
uv run python scripts/train_all.py

# Remoto (submete os 4 para a fila)
uv run python scripts/train_all.py --remote
```

---

### 4. Coletar resultados após execução remota

```python
from clearml import Task

task = Task.get_task(project_name="Transpetro", task_name="autoencoder-B-402E")

# Baixar modelo treinado
model_path = task.artifacts["model_file"].get_local_copy()

# Métricas e threshold
results = task.artifacts["results"].get()
# {'threshold': 0.012, 'train_mse_mean': ..., 'n_anomalies': 150, ...}

# DataFrame com erro de reconstrução por timestamp
scores_df = task.artifacts["test_scores"].get()
# index=datetime, columns=[reconstruction_error, is_anomaly]
```

---

## Configuração por equipamento

O preprocessing é configurável por equipamento em [src/transpetro_modelos/config.py](src/transpetro_modelos/config.py) via `preprocessing_steps`:

```python
# B-402E: filtra períodos desligados (Corrente <= 1), remove transientes de partida
preprocessing_steps = [
    {"step": "filter_running", "column": "Corrente", "threshold": 1.0},
    {"step": "remove_transients", "minutes": 10},
    {"step": "normalize", "method": "standard"},
]

# B-8802B, B-4064A, B-90001A: só normalização
preprocessing_steps = [
    {"step": "normalize", "method": "standard"},
]
```

Steps disponíveis: `filter_running`, `remove_transients`, `normalize` (`standard`/`minmax`/`robust`), `select_features`, `resample`.

Para experimentar com diferentes configurações, basta clonar a task no dashboard ClearML, alterar os hyperparâmetros e enfileirar novamente.

---

## Estrutura

```
src/transpetro_modelos/
  config.py              # metadata dos equipamentos e falhas
  data/
    loading.py           # carrega feather local ou via ClearML Dataset
    preprocessing.py     # pipeline configurável de preprocessing
    splitting.py         # split temporal train/val/test
    upload_datasets.py   # upload dos feather para o ClearML
  models/
    autoencoder.py       # DenseAutoencoder (PyTorch)
  training/
    train.py             # loop de treino com early stopping
    evaluate.py          # cálculo de erro de reconstrução e threshold
scripts/
  upload_data.py         # entry point: upload datasets
  train_equipment.py     # entry point: treinar 1 equipamento
  train_all.py           # entry point: treinar todos
```
