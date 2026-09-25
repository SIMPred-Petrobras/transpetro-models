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
| `drift/monitor.py` | `scripts/monitor_drift.py` | semáforo semanal (M1–M5) + M6 com o `CalibratedKSDetector`; `--make-drift-ref` calibra e grava o `drift_ref.json` no bundle |
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

O estado calibrado usa o mesmo formato do `drift_ref.json` gravado nos bundles
(`monitor_drift.py --make-drift-ref`): `CalibratedKSDetector.from_drift_ref(...)` reproduz
exatamente os disparos do monitor (verificado no B-8802B: 88 de 88 no caso de drift, 3 de 3
na produção).

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
