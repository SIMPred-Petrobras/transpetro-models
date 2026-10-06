# Detecção de mudança de conceito — arquitetura unificada

Junta as duas frentes de detecção de drift do projeto: a biblioteca de detectores com
interface comum (branch `Lara`) e o protocolo de monitoramento/retreino do B-8802B
(`docs/politica_retreino.md`).

## Onde fica cada coisa

Tudo em `src/transpetro_modelos/drift/`; em `scripts/` ficam só os comandos, com os mesmos nomes de antes.

| módulo | comando | papel |
|---|---|---|
| `drift/detectors.py` | — | `BaseDriftDetector` (`update`/`reset`, um ponto por vez) + KS por p-valor, PSI, Page–Hinkley, CUSUM, ADWIN-lite e **`CalibratedKSDetector`** (o adotado) |
| `drift/benchmark.py` | — | atraso até detectar e falsos alarmes contra uma mudança conhecida |
| `drift/monitor.py` | `scripts/monitor_drift.py` | semáforo semanal (M1–M5) + M6 com o `CalibratedKSDetector` + M7 dado congelado; `--make-drift-ref` calibra e grava o `drift_ref.json` no bundle |
| `drift/residuo.py` | — | `ResidualLevelDetector`: mudança de nível de uma temperatura, descontada a estação (resíduo de um modelo de comportamento normal) |
| `drift/report.py` | `scripts/drift_report.py` | relatório de investigação quando o monitor sai do verde |
| `drift/battery.py` | `scripts/battery.py` | bateria de validação de um bundle (aprova/reprova) |
| `drift/retrain.py` | `scripts/retrain_pipeline.py` | retreino com portão humano e de dados |
| — | `scripts/compare_drift_detectors.py` | compara todos os detectores no B-8802B (controle × drift real) |
| — | `scripts/package_monitor.py` | gera o monitor autocontido do pacote de deploy |

**O monitor do deploy é gerado, não copiado.** `deploy_v2/Transpetro/monitor_drift.py` é montado por
`scripts/package_monitor.py` a partir de `drift/monitor.py`, com o `CalibratedKSDetector` embutido,
e não depende da lib (só pandas/numpy/scipy e o `simpred_inference.py` do pacote). O algoritmo existe
em um lugar só. Depois de mudar o monitor ou o detector, rode o gerador;
`python scripts/package_monitor.py --check` acusa se a cópia do deploy ficou desatualizada.

`transpetro_modelos.detector` continua existindo só como atalho para o caminho antigo (o código da
branch `Lara` importa de lá).

## O detector adotado: `CalibratedKSDetector`

KS de duas amostras por sensor (ou sobre o erro), com três escolhas de protocolo:

1. **Limiar do estatístico D, não p-valor.** O p-valor assume amostras independentes; em
   série de sensor autocorrelacionada ele sai minúsculo para diferenças triviais. O limiar de
   cada sensor é o maior D observado entre cada janela da própria referência e a referência
   inteira.
2. **Janela diária** (288 pontos a 5 min), sem sobreposição.
3. **Persistência 3 de 5**: dispara quando 3 dos últimos 5 dias ficam acima do limiar.

Por que "3 de 5" e não "3 dias seguidos": no drift real do B-8802B os dias acima do limiar
vêm intercalados. Com a regra de dias seguidos, o atraso dependia de onde cada dia começa:
de 3,4 a 75,7 dias entre 24 alinhamentos possíveis. Com 3 de 5, de 3,4 a 8,4 dias (mediana
3,9), e zero falso disparo no controle sem drift (jul–dez/2025) em todos os alinhamentos.

A referência é resumida por quantis (`sampling="quantile"`, usado pelo monitor), não por um sorteio: o limite e a
amostra guardada no `drift_ref.json` não dependem de semente nem do tamanho. Na varredura, uma parada de mais de 3
dias zera a persistência (`scan(..., reset_gap="3D")`). Trechos de dado congelado (3+ sensores com o mesmo valor por
12 h+, falha de aquisição) ficam fora da calibração e da varredura e viram o aviso M7.

O estado calibrado usa o mesmo formato do `drift_ref.json` gravado nos bundles
(`monitor_drift.py --make-drift-ref`): `CalibratedKSDetector.from_drift_ref(...)` reproduz
exatamente os disparos do monitor (verificado no B-8802B: 88 de 88 no caso de drift, 3 de 3
na produção).

## Temperaturas: o detector no resíduo

O KS compara um dia com a referência inteira (o ano todo). Em temperatura de mancal, que tem estação forte, todo dia
já parece diferente do ano e o limite vai para perto de 1: no B-8802B, um degrau de −6 °C no mancal LA em 28/03/2026
não gerou disparo robusto. O `ResidualLevelDetector` ajusta na referência uma regressão da temperatura contra as
temperaturas do motor, a corrente e as pressões, e monitora a mediana diária do resíduo (faixa p0,5–p99,5 da
referência, 3 de 5 dias). No B-8802B ele dispara 4 dias depois do degrau, sem disparo em 2025; no B-4064A acha a
mudança pós-reparo em ~2,5 dias e não dispara com referência de só 4 meses (o KS dá 2 falsos, pela estação). Só
serve quando a regressão explica o sensor (no mancal LNA do B-8802B, R² 0,32: fica no KS). No monitor é o M8:
`--make-residual-ref` grava `residual_ref.json` no bundle, e `drift/retrain.py` recalibra os dois detectores em todo
bundle novo.
Análise: `notebooks/drift/mudanca_conceito_B-8802B.ipynb`.

## O que não foi incorporado

`scripts/drift_retrain.py` (branch `Lara`) retreina o modelo automaticamente a cada detecção.
Não entra porque o detector também dispara na **falha real** (a de 2022 do B-8802B foi
detectada assim): o retreino automático colocaria a falha dentro da janela de treino. O fluxo
adotado é detecção → relatório com checklist → confirmação humana → retreino automatizado →
bateria de validação (ver `docs/politica_retreino.md`). O script também depende da versão
reescrita de `scripts/automl.py` daquela branch.

## Duas escalas de persistência

- **Anomalia** (modelo): 15 de 20 leituras de 5 min, cerca de 75 min. Filtra ruído pontual.
- **Mudança de conceito** (M6): 3 de 5 dias. Filtra transiente de operação.

Um evento de horas vira alarme, mas não vira drift.
