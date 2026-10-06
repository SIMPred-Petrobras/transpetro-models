# B-8802B → B-8802B-2025: o que muda para a integração

**Resumo em uma linha:** é o mesmo modelo, atualizado com os dados de 2025–26 — troque a pasta do bundle (`B-8802B/` → `B-8802B-2025/`) e atualize o
`simpred_inference.py` compartilhado. **O contrato de entrada/saída não muda.** Nada de novo para
instalar.

---

## 1. É o mesmo modelo — atualizado com os dados novos

O `B-8802B-2025` **não é uma abordagem diferente**: é o **mesmo modelo do B-8802B, na versão 2**. Continua sendo:

- o **mesmo tipo de modelo** (autoencoder variacional — VAE — que aprende a operação normal e mede o erro de reconstrução);
- os **mesmos 5 sensores** de entrada (Pressão Sucção, Pressão Descarga, Vibração Bomba LA/LNA, Temperatura Bomba LA);
- a **mesma lógica de alarme** (erro de reconstrução → limiar → confirmação por persistência → `normal / atencao / alarme`);
- o **mesmo contrato de integração** (4 passos, mesmos arquivos, mesma saída).

O que foi feito foi **treinar de novo com os dados recentes e afinar a calibração**:

| O que melhorou | Antes (B-8802B, 2022) | Agora (B-8802B-2025) |
|---|---|---|
| Dados de treino | ~6 semanas de 2022 (uma única campanha) | **12 meses de 2025** (todos os regimes de operação do ano) |
| Validação | in-sample | **5 meses nunca vistos (jan–mai/2026)** + falha real de 2022 + falha sintética |
| Hiperparâmetros | AutoML de 2022 | **re-otimizados por AutoML** sobre os dados novos (camadas 128-64-32, latente 16) |
| Limiar | quantil por meta de FP | **média + 6,5·σ do erro em operação normal** (régua interpretável) |
| Confirmação | 6 leituras seguidas | **15 de 20 leituras** (mais robusta a ruído isolado) |
| Pré-processamento | 8 passos | **+1 passo**: máscara de 90 min após manobra de pressão |

**Por que foi necessário.** Após o reparo do equipamento (pós-falha de 2022) e com a operação em uma faixa maior de
regimes, o modelo de 2022 passou a ver o novo "normal" como anomalia (~12% de alarme em 2025–26 — *concept drift*).
Recalibrar só o limiar não resolveria: para zerar o falso positivo no novo normal, o limiar ficaria acima do nível da
própria falha de 2022, e o modelo deixaria de detectá-la. Retreinar com o normal atual restaurou a margem: **~0% de
alarme em 2026 e a falha de 2022 continua detectada com ~2 dias de antecedência**.

Em resumo, para a integração: **é o B-8802B de sempre, com pesos e calibração novos** — por isso a troca é só de pasta.
As duas pastas coexistem de propósito (histórico/comparação). **Em produção, use só o `B-8802B-2025`.**

## 2. O que muda (tabela)

| Item | Antigo `B-8802B/` | **Novo `B-8802B-2025/`** | Impacto na integração |
|---|---|---|---|
| Pasta do bundle | `modelos/model_2022-05-15_2022-07-21_VAE/` | `modelos/model_2025-01-01_2026-08-10_VAE/` | trocar o caminho |
| Script de exemplo | `scripts/b8802b_exemplo.py` | `scripts/b8802b2025_exemplo.py` | mesmo fluxo de 4 passos |
| Arquitetura (VAE) | 5 → [32, 16, 8] → latente 32 | 5 → [128, 64, 32] → latente 16 | nenhum — vem do `model_arch.json` |
| Sensores de entrada | 5 (Pressão Sucção/Descarga, Vibração Bomba LA/LNA, Temperatura Bomba LA) | **os mesmos 5** | nenhum |
| `pipeline.json` | 8 passos | **9 passos** — novo `remove_regime_transients` | módulo atualizado já implementa |
| Limiar de alarme (`threshold`) | 0,4443 | **0,3362** (= μ + 6,5·σ do erro em operação normal) | lido do `alarm.json` |
| Limiar de atenção | 0,3531 | **0,2289** (= μ + 4·σ) | lido do `alarm.json` |
| Persistência | `debounce_consecutive: 6` (6 leituras seguidas) | **`debounce_window: 20`, `debounce_min: 15`** (15 de 20 leituras) | módulo atualizado já implementa |
| Dependências | pandas · numpy · torch · scikit-learn | **as mesmas** (`requirements.txt` inalterado) | nenhum |

> Os valores de limiar **não são comparáveis** entre os dois modelos (escalas de erro diferentes). Sempre leia do
> `alarm.json` do bundle em uso; nunca fixe no código.

## 3. O que NÃO muda

- **Entrada:** CSV com 1ª coluna de timestamp e os sensores nas demais (mesmos nomes de coluna).
- **Os 4 passos:** `carregar_dados → preprocessar → carregar_modelo → prever` — mesmas funções. `prever()` ganhou
  um argumento opcional, `df_bruto`: passe o mesmo DataFrame lido por `carregar_dados` (ver o passo 1 abaixo).
- **Saída de `prever()`:** DataFrame indexado por timestamp com `reconstruction_error`, `is_anomaly`, `severity`
  (`normal` / `atencao` / `alarme`) e a coluna nova **`alerta`**: o alarme que durou 1 hora ou mais. **É a coluna
  `alerta` que vai para a operação.**
- **Formato dos artefatos:** `model_state.pt`, `model_arch.json`, `scaler.pkl`, `clip_bounds.json`, `pipeline.json`, `alarm.json`.

## 4. Passo a passo da migração

1. **Atualize o módulo compartilhado** `Transpetro/simpred_inference.py` (versão desta entrega). Ele ganhou:
   - o passo `remove_regime_transients` no despacho de pré-processamento;
   - persistência **k-de-n** (`debounce_window`/`debounce_min`), **retrocompatível**: bundles antigos que só têm
     `debounce_consecutive` continuam funcionando igual;
   - persistência contada no **tempo** e regra de **1 hora** (ligadas por `persistence_mode` e `min_alert_hours` no
     `alarm.json` deste bundle). Com `prever(..., df_bruto=df)`, os minutos de parada que o pré-processamento
     preenche repetindo o último valor ficam fora do alarme; sem isso, um alarme podia "atravessar" uma parada;
   - o passo `remove_frozen_segments` (dado congelado), disponível para os próximos bundles.
2. **Aponte para o novo bundle**: `Transpetro/B-8802B-2025/modelos/model_2025-01-01_2026-08-10_VAE/`.
3. **Rode o exemplo** para validar o ambiente:
   ```bash
   cd Transpetro/
   python3 B-8802B-2025/scripts/b8802b2025_exemplo.py
   ```
   Saída esperada com o CSV de exemplo (jan/2025 → 10/ago/2026): **139.264 instantes · 30 instantes de alarme em 4
   episódios** (06/12/2025, 17/01/2026, 24/07/2026, 10/08/2026) **· 1 alerta** (17/01/2026 22:30, o único alarme
   de 1 hora ou mais). Se os números baterem, a integração está correta.
4. (Opcional) Compare com o antigo rodando `B-8802B/scripts/b8802b_exemplo.py` sobre o mesmo CSV: ele deve dar
   **~12% de alarme** — é a evidência do drift.

## 5. O novo passo de pré-processamento: `remove_regime_transients`

Mesma ideia do `remove_transients` (que descarta os 90 min após uma **partida** da bomba), mas disparado por
**manobra de processo**: quando a pressão varia bruscamente em 15 min (**> 2,4 bar na sucção** ou **> 4,8 bar na
descarga**), os **90 min seguintes são descartados** — o autoencoder não deve tratar manobra operacional como
anomalia do equipamento. Os parâmetros estão no `pipeline.json`:

```json
{"step": "remove_regime_transients",
 "columns": ["Pressão Sucção", "Pressão Descarga"],
 "deltas": [2.4, 4.8], "minutes": 90, "window": 3}
```

Efeito prático: ~4% do tempo de operação é descartado; falso positivo cai ~40% **sem** alterar a sensibilidade
(falha de 2022 e falha sintética detectadas igual).

## 6. Como interpretar os alarmes do modelo novo

- **`alarme`** só dispara com sinal **sustentado** (15 de 20 leituras de 5 min com a bomba operando acima do
  limiar). Um pico isolado não alarma.
- **`alerta`** é o alarme que durou **1 hora ou mais**, a partir desse ponto: é o que vai para a operação. Uma
  degradação real não some em minutos (a falha de 2022 ficou ~2 dias em alarme), então a regra corta só oscilações
  curtas, ao custo de o alerta sair até 1 hora mais tarde.
- Em 19,5 meses de dados o modelo deu 4 alarmes e **1 alerta: 17/01/2026**, vibração dos dois mancais acima do
  esperado. É uma anomalia real nos sensores a checar com a operação, não uma "falha" confirmada.
- **`atencao`** (limiar μ+4σ) é nível informativo; não deve abrir OS.

## 7. Formato dos dados de entrada (atenção para o online)

O dado 2025–26 veio do IFIX em formato **COV** (uma linha por mudança de valor). O pacote espera a série
**em grade regular de 1 min com "segura o último valor"** (hold-last-value), uma coluna por sensor — o CSV de
exemplo já está assim. Se a integração ler direto do historiador, faça essa reamostragem **antes** de chamar
`carregar_dados`. O `pipeline.json` cuida do resto (reamostra para 5 min, filtra bomba desligada etc.).

## 8. Rollback

Basta voltar o caminho do bundle para `B-8802B/…` — o módulo atualizado é retrocompatível. (Mas lembre que o
antigo dá ~12% de alarme no dado atual.)

## 9. Próximos passos (roadmap, não bloqueia a integração)

- **Monitor de drift**: rodar 1× por semana (`monitor_drift.py`); guia completo, com o fluxo de retreino, em
  **`../MONITORAMENTO.md`**. Recomendação: **não** retreinar automaticamente por período fixo (risco de absorver
  uma degradação lenta no "normal").
- Após **manutenção/reparo** do equipamento, avisar o time de modelos: pode ser necessário recalibrar o limiar.
