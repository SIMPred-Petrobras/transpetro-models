# Notebooks — melhor resultado por equipamento

> Atualizado em 02/jun/2026. ⭐ = notebook com o MELHOR resultado / recomendado do equipamento.

## B-8802B — trinca nas lâminas do acoplamento (06/jul/2022)
- ⭐ **`b8802b/automl_best_model_v4_full.ipynb`** — dense/baseline (task `b10ee0cc`, 12.330 trials).
  Ponto de operação limpo: **p99.7 + debounce 10/12 → 55,6% de detecção (7d) @ 0,00% FP** (zero FP
  precoce/começo da série). Lead ~2 dias. (O "88% @ 0,69%" do ranking era em limiar sensível, com FPs.)
- Apoio: `automl_best_model_v4_quick.ipynb` (quick + ajuste fino), `automl_best_model_v2.ipynb` (histórico).

## B-6511502A — quebra das lâminas do acoplamento (15/mai/2023)
- ⭐ **`b6511502a/automl_best_model_ajustado.ipynb`** — **DENSE, p99.7 + debounce 8/10 →
  68,9% de detecção (3d) @ 0,06% FP, ZERO FP antes do dia 07, lead ~49h.** RECOMENDADO.
- `automl_b6511502a_v2_full.ipynb` — o LSTM que o AutoML escolheu (task `583d0560`) + **ensemble**:
  AND consenso (60% @ 0,00% FP, limpa o cluster de 06/mai) e score combinado (91% @ 0,36%, p/
  detecção máxima). Documenta a lição: **dense > LSTM** (seleção do AutoML era enviesada — ver
  `docs/auditoria_pdm.md` §D).

## B-4064A — roçamento rotor-carcaça (30/ago/2024)
- ⭐ **`b4064a/automl_b4064a_prod.ipynb`** — consolidado:
  - Parte 1 (detector da falha de 2024): dense/load_residual_ma; re-threshold p99.5 →
    **71% @ 0,00% FP** (preset load_residual venceu o baseline).
  - Parte 2 (produção 2025-26): re-baseline no saudável de 2025, z>2.5 + deb 6/8 → **0% FP no ano
    subsequente**; pega recorrência severa (~2d de lead). Alarme de fase tardia (limite físico).
  - Contexto: regime térmico pós-reparo (+20°C no mancal LNA) — ver `docs/analise_b4064a_regime.md`.
- **`b4064a/diagnostico_regime_deploy.ipynb`** — diagnóstico executado para o deploy: confirma que
  o sinal da falha de 2024 é forte no regime antigo (Temp. Bomba LNA: 83,3% @1% FP; Mahalanobis 93,8%),
  mas o bundle pré-reparo não é bom para 2025+ sem re-baseline.

## B-0302C — falha no motor elétrico (30/ago/2024) [FRACO — documentação]
- `b0302c/automl_b0302c_full.ipynb` — LSTM, **30% @ 0,87% FP** (acima do ~0% previsto na triagem,
  mas longe dos bons). Limite: 22 canais de motor zerados no dataset (`docs/analise_b0302c.md`).
- **`b0302c/diagnostico_modelo_fraco.ipynb`** — análise de dados/histogramas: 21 sensores mortos/quase
  mortos; após o preprocessing sobram só 2 colunas; ~66% do regime útil abaixo do cutoff de atividade;
  teto univariado e Mahalanobis **0,0% @1% FP**.

## Transversal
- **`validacao_falhas_sinteticas.ipynb`** — envelope de detecção (falhas sintéticas) dos 3 bons:
  B-8802B (gradual 3-7d → 100%, lead até ~6d), B-6511502A (nível real gradual → 80%),
  B-4064A (+20°C/7d → 80%, ~70h).

## B-4703.24001B — desgaste mancal LNA motor (01/out/2022) [FRACO — documentado]
- `b4703_24001b/automl_b4703_doc.ipynb` — full c/ `--select-by heldout` (task `d7375726`):
  melhor = VAE **0,8% @ ~1% FP held-out** (re-threshold: 2,3% @ 0,16%) — nível de ruído, confirma
  a triagem. Desgaste de rolamento exige espectro/envelope (BPFO/BPFI/BSF); RMS 1min não enxerga.
- **`b4703_24001b/diagnostico_modelo_fraco.ipynb`** — análise de dados/histogramas: depois do
  preprocessing restam só 4,8% das linhas brutas em operação; melhor sensor **0,8% @1% FP** e
  Mahalanobis **0,1% @1% FP**.

## Descartados pela triagem (sem notebook de resultado; treinar só p/ documentar)
- B-402E, B-5401A — instrumentação insuficiente p/ o modo de falha
  (~0-5% @1%FP na triagem; ver `docs/auditoria_pdm.md` e memória do projeto).
- **`b402e/diagnostico_modelo_fraco.ipynb`** — documenta que, depois do filtro de operação, restam
  5,9% das linhas brutas; a janela pré-falha de 7 dias tem só 1 ponto em operação, insuficiente para
  estimar precursor.
- **`b5401a/diagnostico_modelo_fraco.ipynb`** — documenta que, depois do filtro de operação, restam
  9,1% das linhas brutas; melhor sensor **0,5% @1% FP** e Mahalanobis **0,0% @1% FP**.

## Bases interpoladas / não trabalhadas a fundo
- **`b-90001a_interpolated/diagnostico_modelo_fraco.ipynb`** — a config aponta para
  `Dados/B-90001A_interpolated.csv`, que não existe localmente; o notebook usa fallback
  `Dados/B-90001A.feather`. Nesse feather há sinal forte (melhor sensor 81,0% @1% FP; Mahalanobis
  79,9%), então este caso merece trabalho de modelagem quando a base interpolada correta estiver
  disponível.
- **`b-3403c_interpolated/diagnostico_modelo_fraco.ipynb`** — a config aponta para
  `Dados/B-3403C_interpolated.csv`, que não existe localmente; o notebook usa fallback
  `DadosV2/B-3403C_pivoted.feather`. Nesse feather há sinal forte (melhor sensor 77,0% @1% FP;
  Mahalanobis 100,0%), então não parece um caso fraco de dados, e sim um caso pendente de consolidar
  dataset/variante e rodar AutoML.
