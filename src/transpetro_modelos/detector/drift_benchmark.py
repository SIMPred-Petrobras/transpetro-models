"""
drift_benchmark.py
==================================
Roda cada detector de drift ponto-a-ponto sobre uma série temporal e mede
o que interessa na prática: QUANTO TEMPO/AMOSTRAS o detector leva para
disparar depois que um drift de verdade acontece — e quantos falsos alarmes
ele dá antes disso.

Uso típico (com os erros de reconstrução que o automl já calcula):

    from drift_detectors import default_detectors
    from drift_benchmark import run_drift_benchmark

    detectors = default_detectors(reference=train_errors)  # baseline "normal"
    report = run_drift_benchmark(
        series=full_errors,               # pd.Series indexada por timestamp
        detectors=detectors,
        true_drift_time=pd.Timestamp("2024-06-01"),  # início real da falha/drift
    )
    print(report.sort_values("detection_delay_samples"))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transpetro_modelos.detector.drift_detectors import BaseDriftDetector


@dataclass
class DriftBenchmarkResult:
    detector_name: str
    detected: bool
    detection_delay_samples: int | None       # nº de amostras entre o drift real e o alarme
    detection_delay_time: pd.Timedelta | None  # mesma coisa, em tempo real (usa o índice da série)
    first_alarm_index: pd.Timestamp | None
    n_false_alarms_before_drift: int           # alarmes disparados ANTES do drift real (ruído)


def _run_single_detector(
    detector: BaseDriftDetector,
    series: pd.Series,
    true_drift_time: pd.Timestamp,
    name: str | None = None,
) -> DriftBenchmarkResult:
    detector.reset()
    label = name or detector.name

    n_false_alarms_before = 0
    first_alarm_after_idx: pd.Timestamp | None = None
    first_alarm_after_pos: int | None = None
    drift_pos = int(series.index.searchsorted(true_drift_time))

    for pos, (ts, value) in enumerate(series.items()):
        fired = detector.update(value)
        if not fired:
            continue
        if ts < true_drift_time:
            n_false_alarms_before += 1
            detector.reset()  # continua monitorando após o falso alarme
            continue
        # primeiro alarme depois (ou no instante) do drift real -> é o que queremos medir
        first_alarm_after_idx = ts
        first_alarm_after_pos = pos
        break

    if first_alarm_after_idx is None:
        return DriftBenchmarkResult(
            detector_name=label,
            detected=False,
            detection_delay_samples=None,
            detection_delay_time=None,
            first_alarm_index=None,
            n_false_alarms_before_drift=n_false_alarms_before,
        )

    return DriftBenchmarkResult(
        detector_name=label,
        detected=True,
        detection_delay_samples=first_alarm_after_pos - drift_pos,
        detection_delay_time=first_alarm_after_idx - true_drift_time,
        first_alarm_index=first_alarm_after_idx,
        n_false_alarms_before_drift=n_false_alarms_before,
    )


def run_drift_benchmark(
    series: pd.Series,
    detectors: dict[str, BaseDriftDetector],
    true_drift_time: pd.Timestamp,
) -> pd.DataFrame:
    """Roda todos os detectores sobre `series` e retorna um DataFrame
    ordenado do mais rápido (menor delay) para o mais lento. Detectores que
    nunca dispararam vão para o fim (NaN em detection_delay_samples).
    """
    results = [
        _run_single_detector(det, series, true_drift_time, name=name)
        for name, det in detectors.items()
    ]
    df = pd.DataFrame([r.__dict__ for r in results])
    return df.sort_values(
        by=["detected", "detection_delay_samples"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def run_drift_benchmark_multi_event(
    series: pd.Series,
    detectors: dict[str, BaseDriftDetector],
    true_drift_times: list[pd.Timestamp],
) -> pd.DataFrame:
    """Mesma ideia, mas agregando vários eventos de drift conhecidos (ex.:
    vários alarmes/falhas históricas do mesmo equipamento). Reporta a
    mediana do delay por detector, que é mais robusta que a média a um
    evento outlier.
    """
    rows = []
    for name, det in detectors.items():
        delays, false_alarms, n_detected = [], [], 0
        for t in true_drift_times:
            r = _run_single_detector(det, series, t, name=name)
            false_alarms.append(r.n_false_alarms_before_drift)
            if r.detected:
                n_detected += 1
                delays.append(r.detection_delay_samples)
        rows.append({
            "detector_name": name,
            "n_events": len(true_drift_times),
            "n_detected": n_detected,
            "detection_rate": n_detected / len(true_drift_times) if true_drift_times else np.nan,
            "median_delay_samples": float(np.median(delays)) if delays else None,
            "mean_false_alarms_per_event": float(np.mean(false_alarms)) if false_alarms else 0.0,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["detection_rate", "median_delay_samples"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)