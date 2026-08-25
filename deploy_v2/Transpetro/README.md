# Deploy v2 — scripts de inferência autocontidos

Versão do entregável reescrita a pedido do time de engenharia. A lógica de inferência
(carregar dados → pré-processar → carregar modelo → prever) fica num **módulo
compartilhado e autocontido** — `simpred_inference.py` — e cada equipamento tem um
**script fino** que só aponta o seu bundle e chama os 4 passos. **Nada depende da
biblioteca `transpetro_modelos`**.

> Comparar com `deploy/` (v1), onde o exemplo era uma casca fina sobre a lib
> (`from simpred_inference import ...` → `from transpetro_modelos import ...`). Aqui o
> `simpred_inference.py` é local ao pacote de deploy e não importa nada interno nosso.

## Dependências (só bibliotecas públicas)

```
pandas · numpy · torch · scikit-learn
```

Versões testadas em `requirements.txt` (Python 3.12). Nenhum código interno nosso é
instalado. Atenção: `scaler.pkl` é um pickle do `RobustScaler`, então a versão do
scikit-learn precisa ser compatível com a do treino (ver `requirements.txt`).

```bash
# a partir desta pasta (Transpetro/)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Como rodar

```bash
# a partir desta pasta (Transpetro/), com python3 (ou o interpretador da venv acima)
python3 B-8802B/scripts/b8802b_exemplo.py
python3 B-6511502A/scripts/b6511502a_exemplo.py
```

Imprime um resumo (instantes, atenção/alarme, 1º e último alarme) e salva o resultado
completo em `<equip>_inferencia.csv` na pasta do script. Resultado idêntico ao da v1
(mesmo modelo, mesmos pesos — só o formato do arquivo mudou).

## O que mudou no bundle (e por quê)

O modelo continua sendo **o mesmo já treinado e validado** — nada foi retreinado.
Só reempacotamos os arquivos para que abram **sem a nossa lib**:

| v1 (`deploy/`) | v2 (`deploy_v2/`) | motivo |
|---|---|---|
| `model.pt` (módulo PyTorch inteiro) | `model_state.pt` (só os pesos) + `model_arch.json` | o `.pt` inteiro só abre com a lib instalada (o pickle referencia nossas classes). Os pesos + a arquitetura escrita no próprio script abrem em qualquer lugar. |
| `preprocessing.pkl` (objeto da lib) | `scaler.pkl` (RobustScaler do sklearn) + `clip_bounds.json` | idem: o `.pkl` era um objeto da nossa lib. Agora são um objeto sklearn puro + um JSON. |
| `pipeline.json`, `alarm.json` | iguais | já eram dados puros (JSON). |

## Estrutura

```
Transpetro/
├── simpred_inference.py            # MÓDULO compartilhado (VAE + 4 passos), autocontido
├── requirements.txt
├── README.md
└── <EQUIP>/
    ├── dados/<periodo>/data_<inicio>_<fim>_raw.csv   # entrada bruta (exemplo)
    ├── modelos/model_<inicio>_<fim>_VAE/
    │   ├── model_state.pt      # pesos treinados
    │   ├── model_arch.json     # input_dim / encoding_layers / latent_dim
    │   ├── scaler.pkl          # RobustScaler ajustado no treino
    │   ├── clip_bounds.json    # limites de clip por sensor
    │   ├── pipeline.json        # passos de preprocessing (ordem + parâmetros)
    │   └── alarm.json          # model_type, threshold, atenção, debounce, features
    └── scripts/<equip>_exemplo.py   # script FINO: aponta o bundle e chama os 4 passos
```

O script fino importa o módulo da pasta `Transpetro/`; a arquitetura `VAE` e todas as
funções de preprocessing vivem em `simpred_inference.py`, num lugar só.

## Pré-processamento: parte em `sklearn.Pipeline`, parte explícita

Os passos foram separados por natureza:

- **Numéricos por coluna** (`clip` → normalização) → montados como um
  **`sklearn.Pipeline([("clip", Clipper(bounds)), ("scale", RobustScaler)])`** no próprio
  script. É o caso de uso clássico do sklearn: ajustado no treino, só `transform` em dado novo.
  O `Clipper` é um `TransformerMixin` visível no script; o `Pipeline` é montado a partir de
  `clip_bounds.json` + `scaler.pkl` (nada é desserializado de classe interna).
- **Temporais** (`filter_running`, `remove_transients`, `resample`, `ffill`, `select_features`)
  → funções explícitas. **Não** viram transformers do sklearn de propósito: descartam linhas /
  dependem do índice temporal, violando o contrato `n_amostras entra = n_amostras sai` que o
  `Pipeline` assume para alinhar com `y`.

## Contrato de inferência (resumo)

1. **Carregar dados** — CSV com 1ª coluna de timestamp; as demais são os sensores.
2. **Pré-processar** — passos temporais explícitos + `sklearn.Pipeline` (clip+scale),
   reusando `scaler.pkl` e `clip_bounds.json` do treino (nunca reajusta — sem vazamento).
3. **Carregar modelo** — reconstrói o `VAE` (arquitetura no próprio script) e carrega
   `model_state.pt`.
4. **Prever** — erro de reconstrução (MSE) por instante; `severity` = normal / atenção /
   alarme. Alarme exige `debounce_consecutive` pontos consecutivos acima do `threshold`.

Ambos os equipamentos prontos usam **VAE** (por isso `.pt`). Se um dia empacotarmos um
modelo sklearn (OCSVM/IsolationForest/LOF), o arquivo do modelo já é um `.pkl` puro que
abre sem lib — nesse caso o passo 3 vira só `pickle.load`.

## B-8802B-2025 — modelo atualizado (retreino pós-drift), lado a lado com o B-8802B

A pasta `B-8802B-2025/` é o **modelo atualizado** do B-8802B (treinado em 2025, validado em 2026);
a pasta `B-8802B/` (modelo de 2022) foi mantida **intacta para comparação**. Em produção use o
`B-8802B-2025`. Diferenças de pipeline (tudo declarado no `pipeline.json`, nada no código):

- novo passo **`remove_regime_transients`**: após um degrau brusco de pressão (manobra de processo —
  |Δ| > 2,4 bar na sucção ou > 4,8 bar na descarga em 15 min) os 90 min seguintes são descartados,
  como já se faz após partidas (`remove_transients`). Já implementado no `simpred_inference.py`;
- limiar **μ + 6,5·σ** do erro em operação normal (2025) e persistência **15 de 20** leituras
  (`debounce_min`/`debounce_window` no `alarm.json`).
