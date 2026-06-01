# B-4064A — Mudança de regime térmico pós-reparo e falsos positivos no ano seguinte

**Data:** 30/mai/2026
**Equipamento:** B-4064A (dataset `B-4064A-novos`, jan/2024 – mar/2026)
**Falha:** 30/ago/2024 — "Roçamento interno do rotor com a carcaça da bomba"

> Documento criado para **não se perder a informação** sobre a variação de temperatura do mancal
> LNA pós-reparo, que é a causa dos falsos positivos no ano subsequente.

---

## 1. O fato central (a variação de temperatura)

A assinatura da falha de 2024 foi **aquecimento do mancal LNA da bomba**: a `Temperatura Bomba LNA`
subiu ~+20°C (de ~46°C para ~66°C) nos últimos 2 dias antes da falha.

**Após o reparo (operação 2025-2026), o mesmo mancal passou a operar permanentemente mais quente:**

| Sensor | Normal pré-falha (2024) | Pós-reparo (2025-26) | Desvio |
|--------|-------------------------|----------------------|--------|
| **Temperatura Bomba LNA** | 46,6°C | **66,9°C** | **+3,71σ** |
| Temperatura Bomba LA | 33,3°C | 45,3°C | +0,93σ |
| Corrente, pressões, temps de motor, vibração | — | — | estáveis (±0,5σ) |

**Consequência:** o nível térmico da falha de 2024 (~66°C) e o do **normal pós-reparo** (~67°C) são
praticamente **iguais**. Logo, um modelo treinado no normal **pré-falha** marca quase toda a
operação de 2025-2026 como anômala (só porque ficou mais quente) — é a **enxurrada de falsos
positivos** observada no ano seguinte. Não é bug de modelo: é mudança real de ponto de operação.

Comprovação (limiar = p99/99.5 do normal pré-falha, contra a operação de 2025-26):
- `Temperatura Bomba LNA` absoluta: detecta a falha (76% em 3d) **mas dispara em ~78% de 2025-26**.
- Nenhuma feature simples separa "falha-quente" de "pós-reparo-quente" (testado: absoluto,
  taxa-de-variação, diferencial entre mancais, razão temp/carga). O diferencial `LNA − LNA motor`
  dá falha +5,01σ **e** pós-reparo +5,04σ (idênticos).

---

## 2. A solução: um modelo, re-baselinado no presente

Detecção de anomalia é **relativa ao normal de agora**. A saída correta:

- **UM modelo** (não dois simultâneos), baselinado na **operação saudável atual** (2025 pós-reparo),
  não no pré-falha. Ele fica quieto na operação normal e acende quando algo foge do normal **atual**
  — incluindo uma recorrência do roçamento (que gera calor *acima* do nível atual).
- **Re-baselinar a cada manutenção** que mude o regime. Padrão de manutenção preditiva.
- **Feature condicionada à carga** (resíduo `Temp ~ Corrente`): captura "mancal mais quente do que a
  carga explica" e é **invariante à troca de regime de carga** (ver seção 4).

Validação (baseline = união de janelas limpas dos 2 regimes de 2025, robust + debounce 6/8):
**0,0% de falso positivo** em toda a operação saudável de 2025-2026. Problema resolvido.

---

## 3. Regimes operacionais em 2025-2026 (não confundir com falha)

- **Regime de carga ALTO** (~41A): jan–jun/2025 e dez/2025–mar/2026.
- **Regime de carga BAIXO** (~37A): jul–nov/2025 (a "restrição operacional"). Temperaturas
  acompanham a carga (mancais mais frios com menos carga) — **operacional, não falha**.
- **Dropouts do sensor de vibração**: valores ≤0,2 (inclusive negativos, fisicamente impossíveis) —
  **mascarar** (filtrar `Vibração > 0,2`).

---

## 4. ⚠️ Candidato a evento NÃO-reportado — maio/2025 (VERIFICAR COM A MANUTENÇÃO)

A varredura multi-lente (4 métodos independentes) convergiu em **um candidato**: aquecimento
sustentado do mancal LNA da bomba em **fim-abril a maio/2025** (pico 09-13/mai), atingindo ~82°C
(mesmo nível absoluto da falha de 2024), ~5 dias contínuos, **não explicado por carga** e com
vibração estável — **mesma família da falha de 2024**.

**Porém:** condicionado à carga é **leve** (+3 a +5°C acima do previsto, vs **+20°C** na falha real)
e **auto-resolveu** sem recorrer em 10 meses (atípico para degradação progressiva de mancal). É o
**único candidato do ano** (z-máx 2,5, o pico de 2025). Provável evento térmico transitório
(roçamento incipiente que se corrigiu? lubrificação? excursão operacional?). **Verificar se houve
algum evento/intervenção em maio/2025.**

---

## 5. Limite honesto de detecção

A 1h, a operação pós-reparo é **termicamente ruidosa** (resíduo `Temp LNA ~ Corrente`: sMAD ~5,8°C),
o que **eleva o piso de detecção**:

| Severidade da recorrência (acima do normal de carga) | Detecção (z>3, debounce 6/8) |
|------------------------------------------------------|------------------------------|
| +5°C / +10°C | ~0% (no ruído) |
| +15°C | ~6% |
| **+20°C (= falha de 2024)** | ~25% (dispara, porém tarde) |
| +25°C | ~68% |

Ou seja: pega uma recorrência **severa** (como a de 2024), mas **tardiamente**; um evento
leve/precoce fica no piso de ruído. **Detecção precoce confiável de roçamento exigiria vibração de
alta frequência (envelope/FFT)** — limitação de *instrumentação*, não de modelo.

---

## 6. Configuração recomendada (produção)

- `dataset = B-4064A-novos`; **mascarar** `Vibração ≤ 0,2` (dropout).
- **baseline = operação saudável atual** (união dos 2 regimes; ex.: `[2025-02..03] ∪ [2025-09..11-15]`),
  excluindo o evento de maio, transientes de partida e gaps.
- **features de resíduo `Temp ~ Corrente`** (LNA e LA) — invariantes ao regime de carga.
- `normalize: robust` + `clip` + **debounce 6/8**; limiar z≈3 (alarme) / z≈2,5 (vigilância).
- **re-baselinar a cada manutenção** que mude o ponto de operação.

Notebook de validação: `notebooks/b4064a/producao_pos_reparo.ipynb`.
Config no código: entrada `B-4064A-prod` em `src/transpetro_modelos/config.py` (preset `load_residual`).
