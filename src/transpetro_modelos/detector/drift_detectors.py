"""Caminho antigo, mantido por compatibilidade: use transpetro_modelos.drift.detectors."""
from transpetro_modelos.drift.detectors import (  # noqa: F401
    ADWINLiteDetector, BaseDriftDetector, CalibratedKSDetector, CUSUMDriftDetector,
    KSDriftDetector, PageHinkleyDetector, PSIDriftDetector, default_detectors,
)
