# Detecção de mudança de conceito — arquitetura unificada

Junta as duas frentes de detecção de drift do projeto: a biblioteca de detectores com
interface comum (branch `Lara`) e o protocolo de monitoramento/retreino do B-8802B
(`docs/politica_retreino.md`).

## Peças

| peça | onde | papel |
|---|---|---|
| Interface e detectores | `src/transpetro_modelos/detector/drift_detectors.py` | `BaseDriftDetector` (`update`/`reset`, um ponto por vez) + KS por p-valor, PSI, Page–Hinkley, CUSUM, ADWIN-lite e **`CalibratedKSDetector`** |
| Benchmark de atraso | `src/transpetro_modelos/detector/drift_benchmark.py` | mede atraso até detectar (amostras e tempo) e falsos alarmes antes de uma mudança conhecida |
| Comparação no B-8802B | `scripts/compare_drift_detectors.py` | roda todos os detectores da fábrica no erro de reconstrução, controle sem drift × drift real |
| Monitor de produção | `scripts/monitor_drift.py` (cópia em `deploy_v2/Transpetro/`) | semáforo semanal + M6; não depende da lib (roda no ambiente da integração) |
| Relatório, bateria, retreino | `scripts/drift_report.py`, `battery.py`, `retrain_pipeline.py` | o que acontece depois da detecção |

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
