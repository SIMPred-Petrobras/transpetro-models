# B-8802B — modelo sem temperatura (out/2026): o que ele faz em cada situação

Bundle: `modelos/model_2025-01-01_2026-08-10_VAE_sem_temp/`. Fica ao lado do modelo em produção
(`model_2025-01-01_2026-08-10_VAE/`). A troca segue o passo de sombra da política: os dois rodam em paralelo por 4 semanas
e a troca só acontece com o aval do time de modelos. Para rodar o novo, aponte `BUNDLE_DIR` do script de exemplo para a pasta nova.

**O que mudou:**
- **Modelo:** saiu a temperatura do mancal LA. Ficaram as vibrações LA e LNA e as pressões de sucção e descarga. A falha de 2022 está na vibração, e a temperatura só puxava o modelo para perto do limiar quando o mancal esfriava (abr–ago/2026, ~6 °C mais frio que o esperado).
- **Treino:** mesma janela (2025) e mesma regra de alarme (limiar μ+6,5σ, 15 de 20 leituras em 100 min, alerta ≥ 1 h).
- **Monitor de temperatura (M8):** agora avisa **só quando o mancal esquenta** acima do esperado (`"lado": "acima"` no `residual_ref.json`, nos dois bundles). Mancal mais frio não indica falha.

| situação | sem temperatura | atual |
|---|---|---|
| operação normal | não alerta | não alerta |
| mancal LA −8 °C ou +8 °C | não alerta | não alerta |
| mancal real ~6 °C mais frio (abr–ago/2026) | não alerta, score 0,11 do limiar | não alerta, score 0,34 (picos 0,88) |
| episódio de vibração 10–12/01/2026 | **alerta** | não alerta |
| episódio de vibração 17/01/2026 | alerta | alerta |
| falha real de 06/07/2022 | alerta ~50 h antes | ~50 h antes |
| falha simulada (assinatura de 2022) | 39 · 13 · 26 · 10 h antes do fim da rampa | 28 · 13 · 21 · 7 h |
| vibração LA +30% (só um mancal) | não alerta | não alerta |
| vibrações LA e LNA +30% | alerta só depois de ~11 dias | não alerta |
| sensor de vibração zerado | não alerta | não alerta |
| sensor travado | não alerta (monitor M7) | não alerta |

**Bateria de aceitação:** o episódio de 10–12/01/2026 entrou como evento esperado, igual ao 17/01. Alertar nele não conta como alarme falso; perdê-lo reprova.
- **Resultado:** 8 dos 12 candidatos sem temperatura aprovados.
- **Escolhido (`l128-64-32 lr0.001 s2`):** 0% de alarme falso, falha de 2022 2,1 d antes, falha simulada 26 h antes (16 h com meia intensidade).
- **Modelo atual:** reprova só por não alertar em 10–12/01.

**M8 só aquecimento:**
- 0 avisos em 2025–26 (com os dois lados eram 30 dias, todos pelo mancal frio);
- mancal +8 °C acima do esperado: aviso em ~2 dias;
- até ~+6 °C: dentro da faixa normal.

**Escopo do modelo:** detectar a volta da falha registrada (trinca nas lâminas do acoplamento, 06/07/2022), que aparece
como um salto forte na vibração dos mancais.

**Fora do escopo dos dois modelos (alarmes do supervisório / instrumentação):**
- sensor de vibração zerado, travado ou com leitura impossível — alarme de instrumentação (o monitor semanal também
  aponta dado congelado, M7);
- subida lenta e moderada de vibração num mancal (+10% a +50%, degradação gradual) — o modelo não foi feito para isso;
  fica para uma etapa futura.

Gráficos e detalhes: `notebooks/b8802b/validacao_sem_temperatura_b8802b.ipynb` e `notebooks/b8802b/lado_a_lado_sem_temperatura_b8802b.ipynb`.
