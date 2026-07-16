# SIMPred · Manutenção Preditiva — Guia de Integração

> **Para o time de engenharia.** Como colocar os modelos desta pasta para rodar (passo a
> passo), como interpretar os alarmes, como os limiares foram calibrados e qual versão de
> código/treino gerou cada modelo.
>
> Este arquivo fica na **raiz** de `Transpetro/` no Drive, fora das pastas de equipamento.
> Fonte versionada no repositório: `deploy/GUIA_INTEGRACAO_DRIVE.md`. Dúvidas: time de
> dados (Francisco).

---

## 1. Modelos publicados

| Equipamento | Modelo | Antecedência na falha histórica | FP com debounce (janela normal) |
|---|---|---|---|
| **B-8802B** | VAE (autoencoder) | ~1,7 dias (alarme 04/07/2022 · falha 06/07/2022) | 0,06% |
| **B-6511502A** | VAE (autoencoder) | ~5 dias (alarme 10/05/2023 · falha 15/05/2023) | ~0% (limiar elevado de propósito — ver §6) |

---

## 2. O que há na pasta de cada equipamento

```
Transpetro/<EQUIPAMENTO>/
├── metadata.csv               # lista de sensores do equipamento
├── dados/
│   └── <período>/data_<início>_<fim>_raw.csv    # dados brutos usados na validação
├── documentos/                # relatório de resultado e visão geral da abordagem
├── modelos/
│   └── model_<início>_<fim>_<ARQ>/   # ← o "bundle": tudo que a inferência precisa
│       ├── model.pt           # o modelo treinado (PyTorch)
│       ├── preprocessing.pkl  # normalização/recortes ajustados no treino (reuso, sem refit)
│       ├── pipeline.json      # pré-processamento congelado (aplicado automaticamente)
│       └── alarm.json         # limiares, debounce, sensores ← fonte da verdade
└── scripts/
    ├── <equip>-exemplo.py     # exemplo completo: carrega → processa → infere
    ├── simpred_inference.py   # núcleo de inferência (acompanha o bundle)
    └── transpetro_modelos-0.1.0-py3-none-any.whl   # pacote Python (instalar 1×)
```

O nome da pasta do bundle informa a **janela de dados** do treino/validação e a arquitetura —
ex.: `model_2022-05-15_2022-07-21_VAE`. Cada pasta de equipamento é **autossuficiente**:
não depende de nada fora dela.

---

## 3. Passo a passo — instalar e rodar (5 minutos)

**Requisitos:** Python **3.12 ou superior** e `pip`. Não precisa de GPU nem de internet
em produção.

```bash
# 1) (recomendado) criar um ambiente isolado
python3.12 -m venv .venv && source .venv/bin/activate

# 2) instalar o pacote (o wheel está em scripts/ de qualquer equipamento)
pip install Transpetro/B-8802B/scripts/transpetro_modelos-0.1.0-py3-none-any.whl
#    → puxa torch, pandas, scikit-learn e pyarrow automaticamente.
#    Dica p/ máquina sem GPU (instala um torch MUITO menor): rode antes
#    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) rodar o exemplo pronto
python Transpetro/B-8802B/scripts/b8802b-exemplo.py
```

**Saída esperada:** o script imprime o nº de amostras pontuadas e os alertas — que devem se
concentrar perto da data da falha histórica (04–06/jul/2022 no B-8802B). Se isso apareceu,
o ambiente está OK.

---

## 4. Integração programática

```python
import sys; sys.path.insert(0, "Transpetro/B-8802B/scripts")
from simpred_inference import load_bundle, predict

bundle = load_bundle("Transpetro/B-8802B/modelos/model_2022-05-15_2022-07-21_VAE")
result = predict(bundle, dados)   # caminho de um CSV ou um pandas.DataFrame
```

### O que ENTRA

- **Colunas:** exatamente os sensores listados em `alarm.json → "features"` (mesmos nomes).
- **Índice temporal** (timestamp) — a 1ª coluna do CSV é interpretada como datetime.
- **Valores brutos, sem nenhum tratamento.** Não é preciso normalizar, reamostrar nem filtrar:
  o bundle reaplica internamente o mesmo pré-processamento do treino — inclusive **descartar
  sozinho os períodos com o equipamento parado** e leituras inválidas.
- **Histórico mínimo:** o modelo pontua janelas de 24 pontos de 5 min — envie pelo menos
  **~2 h de dados** antes do período que quer pontuar.
- Amostragem irregular (padrão COV do historiador) é aceita.

### O que SAI

`DataFrame` indexado por timestamp, com:

| Coluna | Significado |
|---|---|
| `reconstruction_error` | score de anomalia (float) — quanto maior, mais longe do normal |
| `severity` | `normal` · `atencao` · `alarme` (comparação com os dois limiares) |
| `is_anomaly` | `True` = **alarme confirmado** (limiar de alarme + debounce) — é este campo que deve acionar a operação |

---

## 5. Como o alarme funciona (o essencial em 4 linhas)

1. O modelo aprendeu o **comportamento normal** do equipamento; para cada instante ele calcula
   um **score de anomalia** (erro de reconstrução): "quão diferente do normal está agora?".
2. O score é comparado com **dois limiares**: **atenção** (nível baixo — registrar/observar,
   não abre OS) e **alarme** (nível alto — inspecionar).
3. **Debounce:** o alarme só confirma se o score ficar acima do limiar por
   **6 pontos consecutivos** (6 × 5 min = **30 minutos**) — isso suprime picos isolados de ruído.
4. `is_anomaly = True` é o alarme já com debounce — pronto para consumo.

---

## 6. Calibração dos limiares — de onde vêm os números

Os limiares **não são escolhidos à mão**. A calibração funciona assim:

- Escolhe-se uma **meta de falso positivo**: ex. "aceito que no máximo 1% do tempo de operação
  normal fique acima do limiar".
- O limiar é o **quantil correspondente** do score numa **janela comprovadamente normal**
  (longe de falhas). Meta de 1% → limiar = percentil 99 do score nessa janela.
- Dois níveis: **atenção** com meta ~5% (sensível, para acompanhamento) e **alarme** com meta
  ≤1% (confiável, para acionar inspeção).

Valores em produção (gravados no `alarm.json` de cada bundle — **fonte da verdade**):

| | **B-8802B** | **B-6511502A** |
|---|---|---|
| Limiar de **alarme** | 0,390 (meta FP 1% · medido 1,01%) | 0,632 (meta FP 0,45% · medido 0,45%) |
| Limiar de **atenção** | 0,210 (meta FP 5%) | 0,325 (meta FP 5%) |
| Debounce | 6 pontos (30 min) | 6 pontos (30 min) |
| Janela normal da calibração | 15/05/2022 → 29/06/2022 (6.649 pts) | 20/05/2022 → 18/04/2023 (34.890 pts) |
| FP em validação held-out (fora do treino) | 0,71% | 0,14% |

**Nota (B-6511502A):** o limiar de alarme foi deliberadamente elevado além da meta de 1% para
levar o falso positivo a ~zero. Troca consciente: perde-se um pouco de antecedência
(o 1º alarme passa de ~9 para ~5 dias antes da falha), ganha-se um alarme que quase nunca
dispara em vão.

> ⚠️ **Não editar os limiares manualmente.** Se a taxa de alarme em produção parecer errada,
> avise o time de dados — a recalibração é feita por script, sempre por meta de FP.

---

## 7. Versão do código e proveniência de cada modelo

Rastreabilidade completa de como cada modelo foi treinado:

| | **B-8802B** | **B-6511502A** |
|---|---|---|
| Bundle | `model_2022-05-15_2022-07-21_VAE` | `model_2022-05-17_2023-05-17_VAE` |
| Repositório | `Transpetro-modelos` | `Transpetro-modelos` |
| Branch · commit do treino | `main` · `ccaf308cbd4d2322a17ee55f041859b441f43b81` | `main` · `ccaf308cbd4d2322a17ee55f041859b441f43b81` |
| Script de treino | `scripts/automl.py` (AutoML; seleção por FP held-out, meta FP 1%) | idem |
| Experimento (ClearML, projeto *Transpetro*) | `automl-b8802b-deploy` · id `17881db0e8a044a18909c3db6a084467` | `automl-b6511502a-deploy` · id `6d245d9df75a49f2b7fafec37b80d8f5` |
| Data do treino | 13/06/2026 | 13/06/2026 |
| Empacotado/recalibrado em | 14/06/2026 | 26/06/2026 |
| Pacote de inferência | `transpetro_modelos 0.1.0` (wheel incluído) | idem |

Com o id do experimento, o time de dados reproduz o treino e recupera todos os artefatos.

---

## 8. O contrato — e quando avisar o time de dados

A única responsabilidade do lado da integração é **entregar os sensores certos e consumir
`is_anomaly`**. Toda a lógica do modelo (filtros, limiares, recalibração, retreino) fica do
lado do time de dados. Avise-nos quando:

| Situação | Por quê |
|---|---|
| Sensores **renomeados, removidos ou trocados** no historiador | O contrato de entrada quebra — os nomes devem bater com `alarm.json → features`. |
| **Manutenção de grande porte / reforma** no equipamento | O "normal" muda fisicamente e o modelo precisa ser re-baselinado por nós. |
| Taxa de alarme visivelmente fora do esperado | Indica necessidade de recalibração — não ajuste limiar na ponta. |

---

## 9. FAQ

- **Precisa de GPU?** Não. CPU comum resolve (ver dica do torch CPU no §3).
- **Precisa de internet?** Não. O bundle é autossuficiente; nada é enviado para fora.
- **E se vier um valor faltando (NaN)?** Falhas curtas de coleta são toleradas
  (preenchimento limitado no pré-processamento interno); períodos longos sem dado não são
  pontuados. Se um sensor inteiro sumir, **não use o modelo** — avise o time de dados.
- **Com que frequência rodar?** O modelo trabalha em grade de 5 min; rodar a inferência a cada
  5–30 min dá detecção com folga face à antecedência observada (dias).

---

*Documento do time de dados SimPred · jul/2026 · fonte: `deploy/GUIA_INTEGRACAO_DRIVE.md` (repo `Transpetro-modelos`).*
