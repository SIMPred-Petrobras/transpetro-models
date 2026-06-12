"""
SIMPred — Inferência de detecção de anomalias (núcleo reutilizável)
===================================================================

Este módulo implementa o caminho de inferência de um modelo já treinado,
seguindo as 4 partes do padrão SIMPred:

    1. Carregamento dos dados
    2. Carregamento do modelo
    3. Transformações e processamento
    4. Inferência do modelo

Um "bundle" de produção (uma pasta `model_<inicio>_<fim>_<ARQ>/`) contém:

    bundle/
    ├── model.pt            # modelo (torch.save do módulo) — ou model.pkl (OCSVM/IF/LOF)
    ├── preprocessing.pkl   # PreprocessingArtifacts ajustado no treino (scaler/clip/coefs)
    ├── pipeline.json        # passos de preprocessing congelados do equipamento
    └── alarm.json           # model_type, threshold, debounce, seq_len, features

Dependência: o pacote `transpetro_modelos` precisa estar instalado (é a fonte
única do preprocessing e das arquiteturas de modelo — evita reimplementar a
pipeline e divergir do treino).

Uso programático:
    from simpred_inference import load_bundle, predict
    bundle = load_bundle("model_2022-05-15_2022-07-21_DENSE")
    result = predict(bundle, "data_2022-05-15_2022-07-21_raw.csv")
    # result: DataFrame [reconstruction_error, is_anomaly] indexado por timestamp
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training.evaluate import (
    apply_debounce,
    compute_reconstruction_errors,
    score_test_set_sequence,
)


# ──────────────────────────────────────────────────────────────────────────────
# Bundle
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Bundle:
    """Conteúdo de um modelo de produção, pronto para inferência."""
    model: Any
    preprocessing: Any                 # PreprocessingArtifacts (scaler/clip/coefs do treino)
    pipeline_steps: list[dict]         # passos de preprocessing congelados
    model_type: str                    # dense | lstm | vae | ocsvm | isolation_forest | lof
    threshold: float
    debounce_consecutive: int
    seq_len: int
    features: list[str]


def load_bundle(bundle_dir: str | Path, device: str = "cpu") -> Bundle:
    """Parte 2 — Carregamento do modelo e dos artefatos de inferência."""
    bundle_dir = Path(bundle_dir)
    alarm = json.loads((bundle_dir / "alarm.json").read_text())
    pipeline_steps = json.loads((bundle_dir / "pipeline.json").read_text())

    with (bundle_dir / "preprocessing.pkl").open("rb") as f:
        preprocessing = pickle.load(f)

    model_type = alarm["model_type"]
    if model_type in ("ocsvm", "isolation_forest", "lof"):
        with (bundle_dir / "model.pkl").open("rb") as f:
            model = pickle.load(f)
    else:
        # torch.save(model) salva o módulo inteiro; requer transpetro_modelos instalado.
        model = torch.load(bundle_dir / "model.pt", map_location=device, weights_only=False)
        model.eval()

    return Bundle(
        model=model,
        preprocessing=preprocessing,
        pipeline_steps=pipeline_steps,
        model_type=model_type,
        threshold=float(alarm["threshold"]),
        debounce_consecutive=int(alarm.get("debounce_consecutive", 1)),
        seq_len=int(alarm.get("seq_len", 24)),
        features=list(alarm.get("features", [])),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inferência
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(bundle: Bundle, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parte 3 — Transformações e processamento.

    Reaplica EXATAMENTE o preprocessing do treino: usa `fitted_artifacts` para
    reutilizar o scaler/clip/coefs ajustados (transform, nunca fit) — sem vazamento
    e na mesma escala que o modelo aprendeu.
    """
    df_proc, _, _ = run_preprocessing(
        df_raw,
        bundle.pipeline_steps,
        fitted_artifacts=bundle.preprocessing,
        return_artifacts=True,
        return_report=True,
    )
    return df_proc


def _reconstruction_errors(bundle: Bundle, df_proc: pd.DataFrame, device: str) -> pd.DataFrame:
    """Calcula erro de reconstrução por timestamp conforme o tipo de modelo."""
    mt = bundle.model_type
    if mt in ("ocsvm", "isolation_forest", "lof"):
        # modelos sklearn: score_samples/decision_function já embutido no objeto salvo.
        from transpetro_modelos.training.evaluate import (
            compute_isolation_forest_errors,
            compute_lof_errors,
            compute_ocsvm_errors,
        )
        fn = {
            "ocsvm": compute_ocsvm_errors,
            "isolation_forest": compute_isolation_forest_errors,
            "lof": compute_lof_errors,
        }[mt]
        errors = fn(bundle.model, df_proc)
        return pd.DataFrame({"reconstruction_error": errors}, index=df_proc.index)

    if mt in ("lstm",):
        # score_test_set_sequence atribui o erro ao último timestamp de cada janela.
        seq = score_test_set_sequence(
            bundle.model, df_proc, seq_len=bundle.seq_len, threshold=bundle.threshold, device=device
        )
        return seq[["reconstruction_error"]]

    # dense / vae: erro ponto-a-ponto
    errors = compute_reconstruction_errors(bundle.model, df_proc, device=device)
    return pd.DataFrame({"reconstruction_error": errors}, index=df_proc.index)


def predict(
    bundle: Bundle,
    data: str | Path | pd.DataFrame,
    device: str = "cpu",
    datetime_column: str | None = None,
) -> pd.DataFrame:
    """Pipeline completo de inferência (partes 1→4).

    `data` pode ser um caminho (.csv/.feather) ou um DataFrame já carregado.
    Retorna DataFrame [reconstruction_error, is_anomaly] indexado por timestamp,
    com a flag `is_anomaly` já filtrada por debounce (k-de-n consecutivos).
    """
    # Parte 1 — Carregamento dos dados
    if isinstance(data, (str, Path)):
        df_raw = _load_data(data, datetime_column)
    else:
        df_raw = data.copy()

    # Parte 3 — Transformações e processamento
    df_proc = preprocess(bundle, df_raw)

    # Parte 4 — Inferência do modelo
    scores = _reconstruction_errors(bundle, df_proc, device)
    scores["is_anomaly"] = scores["reconstruction_error"] > bundle.threshold
    if bundle.debounce_consecutive > 1:
        scores = apply_debounce(scores, consecutive=bundle.debounce_consecutive)
    return scores


def _load_data(path: str | Path, datetime_column: str | None) -> pd.DataFrame:
    """Parte 1 — Carregamento dos dados (csv ou feather), índice temporal."""
    path = Path(path)
    if path.suffix == ".feather":
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        col = datetime_column or ("datetime" if "datetime" in df.columns else df.columns[0])
        df = df.set_index(col)
        df.index = pd.to_datetime(df.index)
    return df.sort_index()
