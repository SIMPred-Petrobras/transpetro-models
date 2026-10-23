# B-6511502A — modelo de 11 sensores (out/2026): o que ele faz em cada situação

Modelo para o deploy: `modelos/model_2022-05-20_2023-04-18_VAE_11s/` (o script de exemplo já aponta para ele).
O modelo anterior, de 14 sensores (`model_2022-05-17_2023-05-17_VAE/`), fica na pasta para rollback.

**O que mudou:** saíram a temperatura do mancal LNA do motor (TE-07A) e o deslocamento axial (ZE-50/51). Eles não
carregam a falha de 2023 (que está na vibração do mancal LA do motor) e geravam alarme quando mudavam por outro motivo.
Ficaram as 8 vibrações radiais, a corrente do motor e as pressões de sucção e descarga. Mesma janela de treino do
anterior (20/05/2022 a 17/04/2023), mesma regra de alarme (limiar μ+5σ, 15 de 20 leituras em 100 min, alerta ≥ 1 h).

| situação | modelo de 11 sensores | modelo anterior (14) |
|---|---|---|
| operação normal | não alerta | não alerta |
| mancal LNA do motor +8 °C ou +15 °C | **não alerta** | 13% do tempo em alerta |
| todas as vibrações 30% maiores | alerta em ~13 h | alerta em ~13 h |
| vibração só do mancal LA do motor +10% a +30% | não alerta | não alerta |
| vibração só do mancal LA do motor +50% | alerta só semanas depois | alerta só semanas depois |
| outro regime de operação (descarga −8 bar, corrente −15 A) | 2% do tempo (é caso de retreino) | 1% do tempo |
| deslocamento axial +0,3 mm | não vê (fora do modelo) | não vê |
| sensor de vibração zerado | alerta em ~10 h | alerta em ~2 h |
| sensor travado | não alerta | não alerta |
| falha real de 15/05/2023 | **alerta 48 h antes** | alerta 48 h antes |
| falha simulada (assinatura de 2023, 4 datas) | detecta todas, 10–14 h antes do fim da rampa | detecta todas, 5–11 h antes |
| perda de sucção em 22/07/2022 (sucção ~1 bar) | **alerta** | não alerta |

**Escopo do modelo:** detectar a volta da falha registrada (quebra do acoplamento, 15/05/2023), que aparece como um
salto forte na vibração do mancal LA do motor.

**Fora do escopo do modelo (alarmes do supervisório / instrumentação):**
- sensor zerado, travado ou com leitura impossível — alarme de instrumentação (o monitor semanal também aponta dado
  congelado, M7);
- limites de temperatura dos mancais e de deslocamento axial — alarmes de processo já configurados no supervisório
  (sem o deslocamento axial no modelo, uma falha do mancal de escora fica com esses alarmes);
- subida lenta e moderada de vibração num só mancal (degradação gradual) — o modelo não foi feito para isso; fica para
  uma etapa futura.

O monitor semanal (`monitor_drift.py`) já tem a referência deste modelo (`drift_ref.json`). Detalhes, gráficos e os 12
candidatos: `notebooks/b6511502a/validacao_v2_b6511502a.ipynb` no repositório de modelos.
