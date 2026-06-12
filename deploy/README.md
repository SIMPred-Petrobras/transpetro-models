# Deploy — empacotamento para o Drive SIMPred

Fluxo para publicar um equipamento no Drive no padrão SIMPred
(`Transpetro/<EQUIP>/{metadata.csv, documentos, dados, modelos, scripts}`).

## Conteúdo deste diretório

- **`simpred_inference.py`** — núcleo de inferência reutilizável (carrega bundle → processa
  → infere). É copiado/usado pelos scripts de exemplo de cada equipamento.
- **`Transpetro/`** — saída gerada pelo empacotamento (ignorada no git; é o que sobe pro Drive).

## Equipamentos a publicar

| Equip | Arquitetura | Config de treino | Observação |
|---|---|---|---|
| B-8802B | DENSE | `B-8802B` | resultado bom |
| B-6511502A | DENSE | `B-6511502A` | resultado bom |
| B-4064A | DENSE | `B-4064A-prod` | resultado bom (regime térmico) |
| B-0302C | LSTM | `B-0302C` | fraco — documentado |
| B-4703.24001B | VAE | `B-4703.24001B` | fraco — documentado |

## Passo 1 — Retreinar (gera o `preprocessing.pkl`)

O `preprocessing.pkl` (scaler/clip/coefs) passou a ser salvo só após o ajuste recente, então
os modelos precisam ser retreinados uma vez. Rode no ClearML (remoto):

```bash
uv run python scripts/automl.py --equipment B-8802B       --remote --mode full --max-fp-rate 0.01 --select-by heldout --clearml-task-name automl-b8802b-deploy
uv run python scripts/automl.py --equipment B-6511502A    --remote --mode full --max-fp-rate 0.01 --select-by heldout --clearml-task-name automl-b6511502a-deploy
uv run python scripts/automl.py --equipment B-4064A-prod  --remote --mode full --max-fp-rate 0.01 --select-by heldout --clearml-task-name automl-b4064a-deploy
uv run python scripts/automl.py --equipment B-0302C       --remote --mode full --max-fp-rate 0.01 --select-by heldout --clearml-task-name automl-b0302c-deploy
uv run python scripts/automl.py --equipment B-4703.24001B --remote --mode full --max-fp-rate 0.01 --select-by heldout --clearml-task-name automl-b4703-deploy
```

Cada run salva os artefatos em `results/automl_<EQUIP>/` (localmente, quando rodado localmente)
ou na task do ClearML. Para baixar os artefatos de uma task remota concluída, use o ClearML
(`Task.get_task(...).artifacts[...].get_local_copy()`) e aponte `--artifacts-dir` para a pasta.

## Passo 2 — Empacotar para o Drive

Com os artefatos em mãos (ex.: `results/automl_B-8802B`):

```bash
uv run python scripts/package_for_drive.py --equipment B-8802B --artifacts-dir results/automl_B-8802B
```

Isso gera `deploy/Transpetro/B-8802B/` com:
- `metadata.csv` (sensores) — **confira o `equipment_type`** (hoje preenchido como "Bomba").
- `dados/<periodo>/data_<inicio>_<fim>_raw.csv`
- `documentos/` (overview + análises do equipamento)
- `modelos/model_<inicio>_<fim>_<ARQ>/` (model.pt + preprocessing.pkl + pipeline.json + alarm.json)
- `scripts/<equip>-exemplo.py` (carrega → processa → infere)

Sem `--artifacts-dir`, gera tudo menos a pasta `modelos/` (útil para adiantar estrutura/dados).

## Passo 3 — Validar a inferência localmente

```bash
uv run python deploy/Transpetro/B-8802B/scripts/b8802b-exemplo.py
```

Deve imprimir o nº de amostras pontuadas e os alertas (concentrados perto da falha).

## Passo 4 — Subir para o Drive

Arraste `deploy/Transpetro/<EQUIP>/` para a pasta do Drive, dentro de `Transpetro/`.

## O bundle de inferência

Cada `model_*/` é autossuficiente para inferência, dado o pacote `transpetro_modelos` instalado:

| Arquivo | Conteúdo |
|---|---|
| `model.pt` / `model.pkl` | modelo treinado (módulo PyTorch inteiro / pickle sklearn) |
| `preprocessing.pkl` | scaler, clip_bounds e coefs do resíduo ajustados no treino |
| `pipeline.json` | passos de preprocessing congelados (`pre_split_steps` + preset vencedor) |
| `alarm.json` | model_type, threshold, debounce, seq_len, features, métricas do treino |
