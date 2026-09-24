# Monitoramento semanal do modelo (drift) — guia de operação

Este guia é para o time de integração. Objetivo: detectar cedo quando o modelo ficou **desatualizado**
(mudança de conceito — reparo, novo regime de operação) para o time de modelos agir antes de o alarme perder
o valor. **Não muda nada na inferência** — é um passo a mais, depois dela, usando o CSV que ela já gera.

## O que rodar (1× por semana, ~segundos)

```bash
# 1) a inferência normal (como já é feito) — gera <equip>_inferencia.csv
python3 B-8802B-2025/scripts/b8802b2025_exemplo.py

# 2) o monitor, sobre o resultado dela
python3 monitor_drift.py \
  --inferencia B-8802B-2025/scripts/b8802b2025_inferencia.csv \
  --alarm  B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE/alarm.json \
  --dados  B-8802B-2025/dados/2025_2026/data_2025-01-01_2026-08-10_raw.csv \
  --bundle B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE \
  --png monitor_b8802b.png
```

Troque o caminho de `--dados` pelo CSV bruto do período que estiver monitorando (mesmo formato de entrada da
inferência). `--dados`/`--bundle` são opcionais, mas habilitam os dois indicadores mais importantes (M5 e M6).

## Como ler a saída

A última linha diz o status:

| status | significado | o que fazer |
|---|---|---|
| 🟢 **VERDE** | modelo saudável | nada — arquivar a saída |
| 🟡 **AMARELO** | algo mudou de forma sustentada (não é defeito da bomba, é o *modelo* ficando desatualizado) | **avisar o time de modelos** com a saída do monitor; a inferência continua rodando normalmente |
| 🔴 **VERMELHO** | mudança forte e sustentada — os alarmes do modelo perdem confiabilidade até revisão | avisar o time de modelos **no mesmo dia**; tratar novos alarmes do modelo com ressalva até o retorno |

O monitor também imprime a linha `M6 (KS diário): drift detectado via <sensor>` — esse sensor indica **onde**
a operação mudou (ex.: "via Vibração Bomba LA"); inclua essa informação no aviso.

**Importante:** amarelo/vermelho NÃO significa problema no equipamento — significa que o "normal" mudou em
relação ao que o modelo aprendeu (ex.: após manutenção). Problema no equipamento é o que o **alarme** da
inferência aponta. Por isso o monitor nunca deve silenciar um alarme.

## O que o monitor mede (resumo)

- **alarme % / atenção % semanais** — taxa de disparo do modelo;
- **erro relativo** — o quanto a operação atual está longe do normal aprendido (1× = igual ao treino);
- **cobertura** — semanas com pouco dado válido não contam como "verde";
- **M5 — saturação de faixa** — % do tempo com um sensor fora da faixa em que o modelo foi treinado (ali o
  modelo fica "cego" àquele sensor);
- **M6 — teste estatístico diário (Kolmogorov–Smirnov)** — dispara quando 3 dos últimos 5 dias ficam diferentes da referência; compara cada dia com a referência do treino
  (arquivo `drift_ref.json` dentro do bundle); é o detector mais rápido (~3 dias).

Os limiares e o processo completo (quando recalibrar, quando retreinar, quem decide) estão na política do time
de modelos; para a integração, o contrato é só: **rodar 1× por semana e avisar quando sair do verde**, sempre
informando também: manutenções/intervenções no equipamento, mudanças de faixa de operação e trocas/recalibrações
de sensor — esses eventos mudam a interpretação.

## Avisos de manutenção do próprio monitor
- Quando o time de modelos entregar um **bundle novo**, o `drift_ref.json` novo já vem dentro dele — nada a fazer.
- O monitor exige `scipy` além de pandas/numpy (já listado no `requirements.txt`).
