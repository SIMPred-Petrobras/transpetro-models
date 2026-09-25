"""
Ciclo completo de mudança de conceito, simulado sobre uma série histórica:

    MONITORANDO ──disparo──> QUARENTENA ──fim──> recalibra ──> MONITORANDO (maturando)
         ^                                                          │
         └──── a cada `recalibra_a_cada_dias`, a referência cresce ─┘  até `ref_max_dias`

- MONITORANDO: o CalibratedKSDetector compara cada janela com a referência vigente.
- Disparo: registra o evento (instante e sensores) e entra em QUARENTENA, sem novos disparos.
- QUARENTENA: coleta `quarentena_dias` de operação do comportamento novo. É o ponto em que, na
  política, a operação confirma se é um novo normal (manutenção, regime) ou degradação.
- Recalibração: a referência passa a ser os dados desde o disparo. Enquanto ela tiver menos de
  `ref_max_dias`, é refeita a cada `recalibra_a_cada_dias` com todo o dado acumulado desde o
  disparo (maturação); depois disso fica fixa até o próximo disparo.
- Parada longa (buraco maior que `parada_max_dias` entre amostras): a contagem de persistência
  do detector é zerada, para dias de antes da parada não se somarem aos de depois.

A simulação assume que toda quarentena termina confirmando um novo normal. Em produção essa
confirmação é humana, e uma degradação confirmada NÃO recalibra (docs/politica_retreino.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from transpetro_modelos.drift.detectors import CalibratedKSDetector


@dataclass
class ResultadoCiclo:
    eventos: pd.DataFrame          # tipo, instante, sensores, ref_inicio, ref_fim, n_ref
    d_diario: pd.DataFrame         # D ÷ limiar do pior sensor por janela avaliada, com a fase vigente
    detector_final: CalibratedKSDetector = field(repr=False)


def simular_ciclo(
    serie: pd.DataFrame,
    referencia_inicial: pd.DataFrame,
    *,
    window_size: int = 288,
    k_consecutive: int = 3,
    persistence_window: int | None = 5,
    quarentena_dias: float = 30,
    recalibra_a_cada_dias: float = 30,
    ref_max_dias: float = 365,
    parada_max_dias: float = 3,
    seed: int = 0,
) -> ResultadoCiclo:
    """Roda o ciclo sobre `serie` (índice temporal, uma coluna por sensor)."""
    cols = list(serie.columns)

    def novo_detector(ref: pd.DataFrame) -> CalibratedKSDetector:
        return CalibratedKSDetector(ref[cols], window_size=window_size, k_consecutive=k_consecutive,
                                    persistence_window=persistence_window, seed=seed)

    det = novo_detector(referencia_inicial)
    eventos = [{"tipo": "calibracao_inicial", "instante": referencia_inicial.index.min(), "sensores": None,
                "ref_inicio": referencia_inicial.index.min(), "ref_fim": referencia_inicial.index.max(),
                "n_ref": len(referencia_inicial)}]
    linhas = []

    fase = "monitorando"
    inicio_conceito = None      # instante do último disparo (começo do novo normal)
    proxima_recal = None        # quando refazer a referência durante a maturação
    ultimo_t = None
    X = serie[cols].to_numpy(dtype=float)

    for i, (t, x) in enumerate(zip(serie.index, X)):
        if ultimo_t is not None and (t - ultimo_t) > pd.Timedelta(days=parada_max_dias):
            det.reset()                                   # parada longa: zera a persistência
        ultimo_t = t

        if fase == "quarentena":
            if t - inicio_conceito >= pd.Timedelta(days=quarentena_dias):
                ref = serie[(serie.index >= inicio_conceito) & (serie.index < t)]
                if len(ref) >= 2 * window_size:
                    det = novo_detector(ref)
                    eventos.append({"tipo": "recalibracao", "instante": t, "sensores": None,
                                    "ref_inicio": ref.index.min(), "ref_fim": ref.index.max(), "n_ref": len(ref)})
                    fase = "maturando"
                    proxima_recal = t + pd.Timedelta(days=recalibra_a_cada_dias)
            continue

        if fase == "maturando" and t >= proxima_recal:
            ref = serie[(serie.index >= inicio_conceito) & (serie.index < t)]
            ref = ref[ref.index >= t - pd.Timedelta(days=ref_max_dias)]
            det = novo_detector(ref)
            eventos.append({"tipo": "recalibracao", "instante": t, "sensores": None,
                            "ref_inicio": ref.index.min(), "ref_fim": ref.index.max(), "n_ref": len(ref)})
            if (ref.index.max() - ref.index.min()) >= pd.Timedelta(days=ref_max_dias - recalibra_a_cada_dias):
                fase = "monitorando"                      # referência madura: fica fixa
            else:
                proxima_recal = t + pd.Timedelta(days=recalibra_a_cada_dias)

        fired = det.update(x)
        if not det._buffer and det.last_d is not None:     # uma janela acabou de ser avaliada
            razao = float(np.max(det.last_d / np.asarray(det.d_crit)))
            linhas.append({"instante": t, "d_razao": razao, "fase": fase})
        if fired and fase == "maturando":
            # referência ainda incompleta: o disparo indica variação do novo normal que ela não cobre,
            # não um conceito novo. Antecipa a recalibração com tudo desde o início do conceito.
            sensores = list(det.last_drift_features)
            ref = serie[(serie.index >= inicio_conceito) & (serie.index <= t)]
            ref = ref[ref.index >= t - pd.Timedelta(days=ref_max_dias)]
            det = novo_detector(ref)
            eventos.append({"tipo": "recalibracao_antecipada", "instante": t, "sensores": sensores,
                            "ref_inicio": ref.index.min(), "ref_fim": ref.index.max(), "n_ref": len(ref)})
            proxima_recal = t + pd.Timedelta(days=recalibra_a_cada_dias)
        elif fired:
            eventos.append({"tipo": "deteccao", "instante": t, "sensores": list(det.last_drift_features),
                            "ref_inicio": None, "ref_fim": None, "n_ref": None})
            fase = "quarentena"
            inicio_conceito = t
            det.reset()

    d_diario = pd.DataFrame(linhas).set_index("instante") if linhas else pd.DataFrame(columns=["d_razao", "fase"])
    return ResultadoCiclo(eventos=pd.DataFrame(eventos), d_diario=d_diario, detector_final=det)
