"""Mudança de conceito (drift): detectores, benchmark, monitor semanal, relatório de
investigação, bateria de validação e retreino com portões.

    detectors  interface comum (BaseDriftDetector) e detectores; o adotado é CalibratedKSDetector
    benchmark  atraso de detecção e falsos alarmes contra uma mudança conhecida
    monitor    semáforo semanal (M1–M6) sobre a saída da inferência; gera o drift_ref.json
    report     relatório de investigação quando o monitor sai do verde
    battery    bateria de validação de um bundle (aprova/reprova)
    retrain    pipeline de retreino com portão humano e de dados

Política: docs/politica_retreino.md · arquitetura: docs/deteccao_drift.md
"""
