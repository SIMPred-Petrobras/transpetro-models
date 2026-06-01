# Análise B-0302C — Por que o sinal é fraco demais para detecção

**Data:** 29/mai/2026
**Equipamento:** B-0302C
**Falha:** 30/ago/2024 — "Falha no motor elétrico com aumento de vibração"
**Conclusão:** equipamento **não recomendado** para detecção com os sensores atuais. Mesmo
destino do B-4703.24001B.

---

## 1. Resumo executivo

A base do B-0302C está **limpa** (222 dias, fev–set/2024, sem zeros após remoção de erros, sem
gaps, sem NaN) e a configuração de preprocessamento está **correta**. O problema **não** é dado
nem janela de avaliação — é **cobertura de sensores**.

A falha é no **motor elétrico**, mas os únicos sensores com dados contínuos são:
- `Vibração Mancal Bomba LA` (vibração do mancal da **bomba**, não do motor)
- `Pressão diferencial filtro`

Os outros 22 sensores (corrente, temperatura de enrolamento, vibração do motor, tensão,
desbalanço) estão **zerados** no dataset HISTORIAN. Uma falha de motor não se transmite com
força suficiente ao mancal da bomba, e a subida que existe é modesta e indistinguível das
oscilações normais de operação.

**Teto de detecção sob a constraint operacional (máx. 1% de falso positivo): ~0%.**
Para referência, o B-8802B atinge **57%** de detecção sob a mesma constraint.

---

## 2. Metodologia: triagem "detecção × FP" (reaproveitável)

Antes de investir um treino completo (full run, horas de GPU) em qualquer equipamento novo,
vale rodar esta triagem rápida e independente de modelo:

1. Reproduzir o pré-split (mesmos `pre_split_steps` do AutoML).
2. Definir janela **normal** (ex.: tudo antes de `falha − 20 dias`) e janela **pré-falha**
   (ex.: últimos 3, 5, 7, 30 dias).
3. Fixar o limiar no **percentil 99 da janela normal** (= 1% de falso positivo por construção).
4. Medir a fração da janela pré-falha que ultrapassa esse limiar = **detecção alcançável @1% FP**.
5. Repetir como proxy multivariado com **distância de Mahalanobis** (aproxima o que um
   autoencoder de N features aprenderia).

**Regra de bolso:** se a detecção for **< 10% @ 1% FP**, o AutoML não vai entregar um detector
útil — o sinal não existe no espaço de sensores disponível. Descartar antes do full run.

---

## 3. Evidências (B-0302C, verificado por 6 ângulos)

| Ângulo | Resultado | Interpretação |
|--------|-----------|---------------|
| Tendência semanal | Não-monotônica — semana −1 (Vib 3,89) **não** é a maior; pico em −4 (4,91) | Sem rampa de degradação clara rumo à falha |
| Tendência diária | Subida real porém modesta nos últimos 3 dias (Vib 3,3 → 4,5; PressDif 0,31 → 0,48) | Existe sinal, mas pequeno |
| Separabilidade | 13% (vib) / 22% (pressão) acima de +2σ; **0%** acima de +3σ (vib) | Fraca |
| Teto detecção @1% FP | **0%** (janelas 3–14 d), 5% (30 d) | Sinal não entra no top 1% do normal |
| Curva detecção × FP | mesmo a **10% FP**: só ~20% de detecção | Custo de FP altíssimo para detectar pouco |
| Mahalanobis (proxy AE) | **0% @1% FP** | Estrutura multivariada não ajuda; AE não salva |

**Confundidores que mascaram o sinal:**
- A operação normal já apresenta picos esporádicos de vibração (chega a 6–8), tão altos quanto a
  subida pré-falha — ex.: pico isolado em −9 d (21/ago) e na semana −4 (02/ago).
- A pressão diferencial do filtro tem um pico enorme em −27 d (91% acima do limiar), que é
  **limpeza/troca de filtro**, não a falha do motor.

---

## 4. Recomendações

1. **Não é problema de preprocessamento nem de janela.** Foram testadas janelas pré-falha de 3 a
   30 dias — nenhuma resolve. (Diferente do B-8802B, onde a janela de avaliação *era* o problema.)
2. **Maior impacto possível:** obter os dados **reais dos sensores do motor** (corrente, vibração
   do motor, temperatura de enrolamento). São eles que capturariam uma falha de motor. Vale
   verificar com o fornecedor se existe um export sem esses canais zerados.
3. **Não investir mais tuning** nos 2 sensores atuais.
4. **Task full-mode em execução** (`e24e61398af94e2aa7139f818e041d40`, janelas 30/60): o resultado
   é previsivelmente fraco (~0–5% de detecção @1% FP). Pode ser cancelada para economizar GPU, ou
   mantida apenas para documentar formalmente o descarte.

---

## 5. Reaproveitamento

A triagem det×FP da seção 2 deve ser aplicada aos **próximos candidatos** (B-5401A, B-402E,
B-6511502A, …) **antes** de qualquer full run, priorizando equipamentos cujo modo de falha esteja
coberto pelos sensores disponíveis (ex.: falha de acoplamento/mancal com sensores de vibração no
local certo, como o B-8802B).
