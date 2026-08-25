# B-8802B (modelo 2025) — justificativa técnica dos alarmes em 2025–2026

**Modelo:** autoencoder VAE treinado em operação normal de 2025 · limiar μ+6,5σ · confirmação 15 de 20 leituras ·
máscara de manobra de processo. **Período avaliado:** 01/01/2025 → 10/08/2026 (19,5 meses, 139 mil leituras de
5 min com bomba ligada). **Resultado:** 4 episódios de alarme (65 leituras, 0,05% do tempo); 16 dos 20 meses sem
nenhum alarme.

## Como ler cada episódio

Para cada episódio informamos **o que os sensores fizeram** durante a janela de alarme em relação ao comportamento
normal de 2025 (média μ e faixa p1–p99 de operação normal), e **quanto cada sensor contribuiu** para o erro do modelo.
A conclusão é sempre no condicional: o modelo aponta uma **anomalia real nos sensores**; se ela corresponde a uma
falha, intervenção ou manobra, só a operação pode confirmar. **Pedimos que a operação verifique cada data.**

Referência normal (2025): Pressão Sucção 4,08 bar (p1–p99: 1,6–5,9) · Pressão Descarga 49,3 bar (43,6–56,3) ·
Vibração Bomba LA 2,31 mm/s (0–3,2) · Vibração Bomba LNA 0,89 mm/s (0–1,3) · Temperatura Bomba LA 52,7 °C (42–61,6).

---

## Episódio 1 — 06/12/2025 22:45 → 07/12/2025 01:20 (2 h 40)
**Figura:** `episodio_1_20251206.png`

| Sensor | Normal (μ) | Durante o alarme (média · extremo) | Desvio |
|---|---|---|---|
| **Temperatura Bomba LA** | 52,7 °C | **61,7 °C · 62,1 °C** | **+2,3σ — acima do p99** |
| Pressão Sucção | 4,08 bar | 5,24 · 5,97 bar | +1,4σ (no limite do p99) |
| Pressão Descarga | 49,3 bar | 50,3 · 51,6 bar | +0,4σ |
| Vibrações LA / LNA | 2,31 / 0,89 | 2,26 / 0,79 | normais |

**O que o modelo viu:** erro máximo 0,466 (limiar 0,336); 36% do erro veio da temperatura do mancal LA e 35% da pressão
de sucção.

**Justificativa:** alarme em virtude de **temperatura do mancal da bomba (lado acoplado) acima da faixa normal por quase
3 horas**, com pressão de sucção elevada e vibração normal. **Pode indicar** aquecimento anormal do mancal LA
(lubrificação, refrigeração, carga) ou um regime de bombeio atípico (produto/temperatura diferente).
**Pergunta à operação:** houve alguma ocorrência, troca de produto ou intervenção na noite de 06→07/12/2025?

---

## Episódio 2 — 17/01/2026 21:30 → 22:40 (1 h 15)
**Figura:** `episodio_2_20260117.png`

| Sensor | Normal (μ) | Durante o alarme (média · extremo) | Desvio |
|---|---|---|---|
| **Vibração Bomba LA** | 2,31 mm/s | **3,60 · 3,87 mm/s** | **+2,6σ — acima do p99** |
| **Vibração Bomba LNA** | 0,89 mm/s | **1,28 · 1,49 mm/s** | **+1,9σ — acima do p99** |
| Pressão Descarga | 49,3 bar | 52,9 bar | +1,4σ |
| Temperatura Bomba LA | 52,7 °C | 47,3 °C | −1,4σ |

**O que o modelo viu:** erro máximo 0,530 (o mais alto dos quatro); 74% do erro veio das **duas vibrações** (38% LNA, 36% LA).

**Justificativa:** alarme em virtude de **vibração elevada simultaneamente nos dois mancais da bomba**, ambos acima
do p99 da operação normal, por mais de 1 hora, com descarga acima do usual. Essa é a assinatura mais próxima de um
problema mecânico entre os quatro episódios. **Pode indicar** desbalanceamento, desalinhamento, folga ou operação fora
do ponto (cavitação/recirculação). Detalhe relevante na figura: a vibração subiu gradualmente ao longo de ~3 h (18h → 21h) e **voltou ao normal às ~23:30, no mesmo instante em que a pressão de descarga caiu de 52,9 para 46,8 bar** — ou seja, o desvio acompanhou um ponto de operação de alta pressão. Se esse regime se repetir, a vibração tende a voltar. **Pergunta à operação:** o que ocorreu na noite de 17/01/2026? Houve manobra,
partida de bomba em paralelo ou ruído/vibração percebidos? **Prioridade de verificação: alta.**

---

## Episódio 3 — 24/07/2026 14:35 → 14:55 (25 min)
**Figura:** `episodio_3_20260724.png`

| Sensor | Normal (μ) | Durante o alarme (média · extremo) | Desvio |
|---|---|---|---|
| Temperatura Bomba LA | 52,7 °C | 44,2 · 43,9 °C | −2,2σ |
| Vibração Bomba LA | 2,31 mm/s | 1,99 · 1,83 mm/s | −0,7σ |
| Pressão Sucção | 4,08 bar | 5,54 bar | +1,8σ |

**Justificativa:** alarme curto (25 min) em virtude de **mancal mais frio que o normal com vibração baixa e sucção alta** —
combinação típica de **operação logo após parada/religamento ou mudança de regime**, não de degradação (nenhum sensor de
equipamento subiu). **Classificação sugerida: operacional / baixa prioridade.** Vale confirmar se houve parada nesse dia.

---

## Episódio 4 — 10/08/2026 12:45 → 16:25 (3 h 45) — último dia dos dados
**Figura:** `episodio_4_20260810.png`

| Sensor | Normal (μ) | Durante o alarme (média · extremo) | Desvio |
|---|---|---|---|
| **Temperatura Bomba LA** | 52,7 °C | **43,2 · 34,9 °C** | **−2,5σ — abaixo do p1** |
| **Vibração Bomba LNA** | 0,89 mm/s | 1,05 · **1,37 mm/s** | +0,8σ (**pico acima do p99**) |
| Pressão Sucção | 4,08 bar | 3,27 · **2,72 bar** | −1,0σ |
| Pressão Descarga | 49,3 bar | 48,3 · 44,3 bar | −0,4σ |
| Vibração Bomba LA | 2,31 mm/s | 1,90 mm/s | −0,8σ |

**O que o modelo viu:** erro máximo 0,558; 35% do erro veio da temperatura (anormalmente baixa), 20% da vibração LNA,
20% da vibração LA. No mesmo dia, fora da janela de alarme, a pressão de descarga registrou pico de 73 bar (normal < 56).

**Justificativa:** alarme em virtude de **condição operacional atípica e sustentada por quase 4 horas**: mancal muito
frio (34,9 °C), **pressão de sucção baixa** (2,7 bar) e **vibração do mancal LNA subindo acima do p99** — combinação
compatível com **restrição/deficiência de sucção (risco de cavitação)** ou partida em condição anormal. **Como a série
termina neste dia, não sabemos o que aconteceu depois.** **Pergunta à operação (prioritária):** o que ocorreu com o
B-8802B em 10/08/2026 e nos dias seguintes? Houve parada, manutenção ou alarme de processo?

---

## Síntese para a reunião

| # | Data | Duração | Sensores em desvio | Leitura | Prioridade |
|---|---|---|---|---|---|
| 1 | 06–07/12/2025 | 2 h 40 | Temp. mancal LA > p99 | possível aquecimento do mancal | média |
| 2 | 17/01/2026 | 1 h 15 | Vibração LA **e** LNA > p99 | possível problema mecânico | **alta** |
| 3 | 24/07/2026 | 25 min | mancal frio, sucção alta | operacional (pós-parada) | baixa |
| 4 | 10/08/2026 | 3 h 45 | mancal frio, sucção baixa, vib. LNA > p99 | possível deficiência de sucção; **fim dos dados** | **alta** |

**Mensagem:** o modelo não gerou alarme em 99,95% do tempo de operação e, quando gerou, **cada alarme corresponde a um
desvio físico mensurável nos sensores**. Não afirmamos que são falhas — pedimos que a operação verifique as quatro datas.
Se algum episódio corresponder a um evento não registrado, o modelo o detectou sem aviso prévio.
