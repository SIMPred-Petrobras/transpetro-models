# Auditoria de PdM (turbocompressores) + ajustes implementados

**Data:** 30/mai/2026 · **Branch:** `feat/pdm-improvements` (criada a partir de `feat/improved-detection`)

Revisão multi-agente (metodologia, cobertura sensor↔falha, regimes/alarme, features) + ajustes.
**Todos os ajustes são OPT-IN: sem ativá-los, o comportamento é idêntico ao anterior.**

---

## Parte 1 — Erros/riscos encontrados (auditoria)

### Metodologia (verificados no código)
1. **FP medido in-sample** [ALTA] — a janela "normal" que mede o FP cai majoritariamente DENTRO do
   train e o threshold é calibrado nos erros do train. Verificado: B-8802B → **97% da janela normal
   está no período de train**. ⇒ o `normal_alert_rate` reportado (ex.: 0,97%) é **otimista**.
   *Mitigação implementada:* nova métrica `val_fp_rate_heldout` (FP medido só na validação, held-out).
2. **Seleção de modelo no test** [ALTA] — o melhor trial é escolhido por `prefailure_alert_rate`
   (janela no test split) entre milhares de trials ⇒ overfit ao ÚNICO evento de falha. O "57%/77%"
   é estimador enviesado para cima. *Mitigação parcial:* reportar `val_fp_rate_heldout`; LOEO (ver Parte 3).
3. **CUSUM e `find_threshold_for_fp_rate` eram CÓDIGO MORTO** [ALTA] — definidos em evaluate.py mas
   nunca chamados pelo runner. *Implementado:* CUSUM agora é política de alarme opt-in no grid.
4. **Threshold global, não por-regime** [ALTA] — operação multi-regime (B-4064A: alto ~41A / baixo
   ~37A) dispara FP nas trocas. *Mitigação:* resíduo condicionado à carga (add_load_residual);
   threshold por-regime fica como proposta (Parte 3, não implementado p/ segurança).
5. **`resample` usava `.last()`** [ALTA] — descartava picos de vibração na janela. *Implementado:*
   `resample` aceita `extra_aggs` (max/rms/...) opt-in.
6. **Resíduo condicionado só no B-4064A** [ALTA] — `add_load_residual` é geral; aplicar a outros é
   opt-in via preset (B-4064A-prod é o template).
7. **Sazonalidade/ambiente não removida** [MÉDIA]; rolling/diff atravessa gaps operantes e perde
   borda entre splits [BAIXA]; `remove_negatives` muta in-place [BAIXA]. (não corrigidos ainda)

### Cobertura sensor↔modo-de-falha (turbomáquinas)
- **Limite físico transversal** [ALTA]: NENHUM equipamento tem espectro/FFT/envelope; cadência máx
  60s. ⇒ falhas de **banda estreita** (rolamento BPFO/BPFI/BSF, desalinhamento 2×, desbalanceamento
  1×, folga, rub incipiente) são **indetectáveis precocemente**. Só se pega o que eleva a energia
  GLOBAL de vibração (rub severo, acoplamento avançado) — e tarde. Explica B-4703 e B-0302C.
- B-0302C: 22 canais de motor zerados (falha de motor sem dados de motor).
- B-6511502A: temp enrolamento/tensão zeradas; instrumentação de óleo/selagem descartada no select_features.
- Falta deslocamento/posição AXIAL na maioria (turbomáquinas com mancal de empuxo).

---

## Parte 2 — Ajustes IMPLEMENTADOS (opt-in) e como reverter

> **Reverter é trivial:** todos os defaults preservam o comportamento antigo. Basta NÃO usar as novas
> opções. Além disso, tudo está na branch `feat/pdm-improvements` — `git checkout feat/improved-detection`
> volta ao estado anterior. As mudanças ainda não foram commitadas.

### (A) `resample` com `extra_aggs` — recupera picos de vibração
`src/transpetro_modelos/data/preprocessing.py`. Default `agg="last", extra_aggs=None` = inalterado.
Uso (no `pre_split_steps` de um config):
```python
{"step": "resample", "freq": "5min", "extra_aggs": {"Vibração Bomba LNA": ["max", "rms"]}}
```
Adiciona colunas `Vibração Bomba LNA__max` / `__rms` (o `select_features` seguinte deve incluí-las).
**Reverter:** remover o `extra_aggs` do step.

### (B) `val_fp_rate_heldout` — FP honesto (held-out)
`src/transpetro_modelos/training/automl.py` (run_trial). Coluna nova no resultado de cada trial:
FP medido só na janela de VALIDAÇÃO (fora do train), com o mesmo debounce. Aditivo, **não altera o
ranking**. Use-a para comparar com o `normal_alert_rate` (in-sample) e ver quão otimista ele é.
**Reverter:** nada a fazer (é só uma coluna extra); pode ignorá-la.

### (C) Política de alarme CUSUM — opt-in no grid
`evaluate.py` (cusum_anomaly_score aceita mu/sigma do train), `automl.py` (TrialConfig.alarm_policy /
cusum_k / cusum_h; run_trial recalcula is_anomaly por CUSUM calibrado no train), `scripts/automl.py`
(`--alarm-policies`). Default `threshold` = inalterado. Uso:
```bash
uv run python scripts/automl.py --equipment B-8802B --remote --mode full \
  --alarm-policies threshold cusum --thresholds 95 --max-fp-rate 0.01 \
  --clearml-task-name "automl-b8802b-cusum-teste"
```
OBS: o CUSUM ignora o `threshold_percentile` → restrinja `--thresholds` (ex.: um só) para não inflar
a grade. `cusum_h` precisa ser TUNADO por equipamento (h alto = menos sensível). Hoje o default h=5
dispara muito em série curta. **Reverter:** não passar `--alarm-policies` (ou passar só `threshold`).

### (D) Seleção por FP held-out (`--select-by heldout`) + desempate consistente
Motivado por caso REAL: no B-6511502A, o AutoML salvou o LSTM como "best" (100% @ 0,87% in-sample),
mas out-of-sample o LSTM tem um cluster de FP (06/mai) inseparável da falha — o DENSE generaliza
melhor (69% @ 0,06%, 0 FP pré-07). E no B-4064A o trial escolhido VIOLAVA a constraint honesta
(held-out 1,51% > 1%). Dois bugs corrigidos:
1. `--select-by heldout` (scripts/automl.py + rank_results(fp_column)): constraint e ordenação usam
   `val_fp_rate_heldout` (FP na validação) em vez do in-sample; fallback p/ in-sample se ausente.
   Default `insample` = comportamento anterior. **Reverter:** não passar o flag.
2. Desempate do "best" no loop agora = mesmo critério do ranking (prefailure desc → FP asc); antes
   o PRIMEIRO a empatar vencia (por isso o LSTM foi salvo com o VAE acima dele no ranking).
**Ressalva:** com 1 falha por equipamento, nenhuma métrica substitui a validação out-of-sample no
notebook (re-threshold + comparação top-N) — o flag reduz o risco, não o elimina.

---

## Parte 3 — Propositadamente NÃO implementado (para próxima rodada, por segurança/escopo)

- **Threshold por-regime** (clusterizar normal por estado operacional) — mudança no core de scoring,
  maior risco; deixado como proposta.
- **Leave-one-equipment-out (LOEO)** — limitado porque equipamentos têm conjuntos de sensores
  DIFERENTES (B-8802B=5 features, B-6511502A=14) → não dá para transferir o mesmo modelo; só dá para
  validar transferência de hiperparâmetro (percentil). Fica como script futuro.
- **Dessazonalização** (incluir T ambiente no resíduo) — exige sensor de ambiente.
- **Generalizar o resíduo como padrão** a todos os equipamentos — mantido opt-in para não degradar os
  que já funcionam (B-8802B, B-6511502A).

## Resumo dos arquivos alterados (branch feat/pdm-improvements)
- `src/transpetro_modelos/data/preprocessing.py` — resample(agg, extra_aggs)
- `src/transpetro_modelos/training/evaluate.py` — cusum_anomaly_score(mu, sigma)
- `src/transpetro_modelos/training/automl.py` — TrialConfig (alarm_policy/cusum_*), run_trial (CUSUM +
  val_fp_rate_heldout), build_trials (alarm_policies)
- `scripts/automl.py` — flag `--alarm-policies`
- (já existentes desta linha de trabalho) config.py B-4064A-prod + add_load_residual; docs/.
