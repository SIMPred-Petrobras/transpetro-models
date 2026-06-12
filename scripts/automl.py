"""
AutoML para Detecção de Anomalias - VERSÃO CORRIGIDA (baseada no seu código)
==================================
Modelos : Dense Autoencoder | LSTM Autoencoder | One-Class SVM | Isolation Forest
Seleção : Score composto balanceado com penalização de falsos positivos
Busca   : Grid search declarativo via build_trials()

Uso rápido:
    python scripts/automl_anomaly_v3.py --equipment MEQ-01 --local-data --mode quick

Uso extensivo (1-2 dias):
    python scripts/automl_anomaly_v3.py --equipment MEQ-01 --remote --queue gpu --mode extensive
    
Uso com constraint de FP (máx 1%):
    python scripts/automl_anomaly_v3.py --equipment MEQ-01 --mode extensive --max-fp-rate 0.01
"""

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clearml import Task
from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import PreprocessingArtifacts, run_preprocessing
from transpetro_modelos.data.splitting import temporal_split
from transpetro_modelos.models.autoencoder import DenseAutoencoder, LSTMAutoencoder
from transpetro_modelos.training.evaluate import (
    apply_debounce,
    compute_balanced_score,
    compute_ocsvm_errors,
    compute_reconstruction_errors,
    compute_reconstruction_errors_sequence,
    determine_threshold,
    fit_ocsvm,
    score_ocsvm_set,
    score_test_set,
    score_test_set_sequence,
    fit_isolation_forest,
    compute_isolation_forest_errors,
    score_isolation_forest_set
)
from transpetro_modelos.training.train import (
    make_dataloader,
    make_sequence_dataloader,
    train_autoencoder,
)


ModelType = Literal["dense", "lstm", "ocsvm", "iforest"]
PresetName = str
ModeType = Literal["quick", "full", "extensive"]


# ════════════════════════════════════════════════════════════════
# TrialConfig
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrialConfig:
    """Configuração imutável de um trial de AutoML."""
    val_start: datetime | None
    preset: str
    model: ModelType
    threshold_percentile: float
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10
    dense_layers: tuple[int, ...] | None = None
    dropout: float = 0.0
    seq_len: int = 24
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    ocsvm_nu: float = 0.05
    ocsvm_gamma: str | float = "scale"
    iforest_contamination: float = 0.05
    iforest_n_estimators: int = 100
    debounce_consecutive: int = 1

    def __post_init__(self):
        """Valida configuração."""
        if self.model not in {"dense", "lstm", "ocsvm", "iforest"}:
            raise ValueError(f"Modelo inválido: {self.model}")
        if not 0 < self.threshold_percentile <= 100:
            raise ValueError(f"threshold_percentile inválido: {self.threshold_percentile}")

    def label(self) -> str:
        """Gera identificador único e legível."""
        vs = self.val_start.strftime("%Y-%m-%d") if self.val_start else "auto"
        parts = [vs, self.preset, self.model, f"p{self.threshold_percentile:g}"]

        if self.model == "dense":
            parts.extend([f"lr{self.learning_rate:g}", f"bs{self.batch_size}"])
            if self.dense_layers:
                layers_str = "-".join(str(d) for d in self.dense_layers)
                parts.append(f"arch_{layers_str}")

        elif self.model == "lstm":
            parts.extend([
                f"seq{self.seq_len}",
                f"hid{self.lstm_hidden_dim}",
                f"lay{self.lstm_num_layers}",
            ])

        elif self.model == "iforest":
            parts.extend([
                f"cont{self.iforest_contamination:g}",
                f"trees{self.iforest_n_estimators}"
            ])
        else:  # ocsvm
            gamma_str = f"{self.ocsvm_gamma:g}" if isinstance(self.ocsvm_gamma, float) else self.ocsvm_gamma
            parts.extend([f"nu{self.ocsvm_nu:g}", f"gamma_{gamma_str}"])

        if self.debounce_consecutive > 1:
            parts.append(f"deb{self.debounce_consecutive}")

        return "__".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Converte para dict serializável."""
        d = asdict(self)
        d["val_start"] = self.val_start.strftime("%Y-%m-%d") if self.val_start else None
        d["dense_layers"] = (
            ",".join(str(v) for v in self.dense_layers)
            if self.dense_layers is not None
            else "auto"
        )
        d["ocsvm_gamma"] = str(self.ocsvm_gamma)
        return d


# ════════════════════════════════════════════════════════════════
# Model Training & Scoring
# ════════════════════════════════════════════════════════════════

def train_model(
    model_type: ModelType,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
    *,
    dense_layers: list[int] | None = None,
    seq_len: int = 24,
    lstm_hidden_dim: int = 64,
    lstm_num_layers: int = 2,
    batch_size: int = 256,
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    logger=None,
    ocsvm_nu: float = 0.05,
    ocsvm_gamma: str | float = "scale",
    iforest_contamination: float = 0.05,
    iforest_n_estimators: int = 100,
) -> tuple[Any, float]:
    """Treina modelo e retorna (model, val_loss)."""
    if model_type == "ocsvm":
        clf = fit_ocsvm(train_df, nu=ocsvm_nu, gamma=ocsvm_gamma)
        return clf, 0.0

    if model_type == "iforest":
        clf = fit_isolation_forest(
            train_df,
            contamination=iforest_contamination,
            n_estimators=iforest_n_estimators,
        )
        return clf, 0.0

    n_features = train_df.shape[1]

    if model_type == "lstm":
        model = LSTMAutoencoder(
            input_dim=n_features,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            seq_len=seq_len,
        ).to(device)
        train_loader = make_sequence_dataloader(
            train_df, seq_len=seq_len, batch_size=batch_size, shuffle=True, device=device
        )
        val_loader = make_sequence_dataloader(
            val_df, seq_len=seq_len, batch_size=batch_size, shuffle=False, device=device
        )
    else:  # dense
        model = DenseAutoencoder(
            input_dim=n_features,
            encoding_layers=list(dense_layers) if dense_layers else None,
        ).to(device)
        train_loader = make_dataloader(
            train_df, batch_size=batch_size, shuffle=True, device=device
        )
        val_loader = make_dataloader(
            val_df, batch_size=batch_size, shuffle=False, device=device
        )

    model, best_val_loss = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        logger=logger,
        return_val_loss=True,
    )

    return model, float(best_val_loss)


def score_full(
    model: Any,
    model_type: ModelType,
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    threshold_percentile: float,
    device: str,
    *,
    seq_len: int = 24,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """Calcula threshold e aplica scores."""
    if model_type == "iforest":
        train_errors = compute_isolation_forest_errors(model, train_df)
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        scores_df = score_isolation_forest_set(model, full_df, threshold)
        return scores_df, threshold, train_errors

    if model_type == "ocsvm":
        train_errors = compute_ocsvm_errors(model, train_df)
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        scores_df = score_ocsvm_set(model, full_df, threshold)
        return scores_df, threshold, train_errors

    if model_type == "lstm":
        train_errors = compute_reconstruction_errors_sequence(
            model, train_df, seq_len=seq_len, batch_size=batch_size, device=device
        )
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        scores_df = score_test_set_sequence(
            model, full_df, seq_len=seq_len, threshold=threshold,
            batch_size=batch_size, device=device,
        )
        return scores_df, threshold, train_errors

    # dense
    train_errors = compute_reconstruction_errors(
        model, train_df, batch_size=batch_size, device=device
    )
    threshold = determine_threshold(train_errors, percentile=threshold_percentile)
    scores_df = score_test_set(
        model, full_df, threshold=threshold, batch_size=batch_size, device=device
    )
    return scores_df, threshold, train_errors


# ════════════════════════════════════════════════════════════════
# Grid Builder
# ════════════════════════════════════════════════════════════════

def build_trials(
    equipment_id: str,
    *,
    mode: ModeType = "full",
    models: list[ModelType] | None = None,
    presets: list[PresetName] | None = None,
    thresholds: list[float] | None = None,
    val_start_dates: list[datetime | None] | None = None,
    dense_layers: list[tuple[int, ...] | None] | None = None,
    dense_lrs: list[float] | None = None,
    batch_sizes: list[int] | None = None,
    weight_decays: list[float] | None = None,
    seq_lens: list[int] | None = None,
    lstm_hidden_dims: list[int] | None = None,
    lstm_num_layers: list[int] | None = None,
    ocsvm_nus: list[float] | None = None,
    ocsvm_gammas: list[str | float] | None = None,
    iforest_contaminations: list[float] | None = None,
    iforest_n_estimators: list[int] | None = None,
    debounce_consecutives: list[int] | None = None,
    epochs: int = 100,
    patience: int = 15,
) -> list[TrialConfig]:
    """Gera grade de trials otimizada para 1-2 dias."""
    config = EQUIPMENT_CONFIGS[equipment_id]
    available_presets = (
        list(config.preprocess_presets.keys())
        if getattr(config, "preprocess_presets", None)
        else ["baseline"]
    )
    default_val_starts: list[datetime | None] = (
        [config.val_start_date] if getattr(config, "val_start_date", None) else [None]
    )

    if mode == "quick":
        _models = models or ["dense", "ocsvm", "iforest"]
        _presets = presets or available_presets[:1]
        _thresholds = thresholds or [95.0]
        _val_starts = val_start_dates or default_val_starts
        _layers = dense_layers or [None]
        _lrs = dense_lrs or [1e-3]
        _batches = batch_sizes or [256]
        _weight_decays = weight_decays or [1e-5]
        _seq_lens = seq_lens or [24]
        _hidden = lstm_hidden_dims or [64]
        _nlayers = lstm_num_layers or [2]
        _nus = ocsvm_nus or [0.05]
        _gammas = ocsvm_gammas or ["scale"]
        _iforest_conts = iforest_contaminations or [0.05]
        _iforest_trees = iforest_n_estimators or [100]
        _debounces = debounce_consecutives or [1]
        _epochs, _patience = 20, 5

        '''    else:  # full
        _models = models or ["dense", "lstm", "ocsvm", "iforest"]
        _presets = presets or available_presets
        _thresholds = thresholds or [90.0, 95.0, 97.5, 99.0]
        _val_starts = val_start_dates or default_val_starts
        
        _layers = dense_layers or [
            None, (64, 32, 16), (128, 64, 32),
            (256, 128, 64), (128, 64, 32, 16),
        ]
        _lrs = dense_lrs or [1e-3, 5e-4, 1e-4]
        _batches = batch_sizes or [128, 256, 512]
        _weight_decays = weight_decays or [0, 1e-5]
        
        _seq_lens = seq_lens or [12, 24, 48]
        _hidden = lstm_hidden_dims or [32, 64, 128]
        _nlayers = lstm_num_layers or [1, 2, 3]
        
        _nus = ocsvm_nus or [0.005, 0.01, 0.05, 0.1]
        _gammas = ocsvm_gammas or ["scale", "auto", 0.01]
        _iforest_conts = iforest_contaminations or [0.005, 0.01, 0.05]
        _iforest_trees = iforest_n_estimators or [100, 200, 500]
        _debounces = debounce_consecutives or [1, 4, 6]'''

    elif mode == "balanced":
        _models = models or ["dense", "lstm", "ocsvm", "iforest"]
        _presets = presets or available_presets[:2]
        _thresholds = thresholds or [95.0, 97.5, 99.0]
        _val_starts = val_start_dates or default_val_starts
        
        _layers = dense_layers or [
            None, (64, 32), (128, 64), (256, 128)
        ]
        _lrs = dense_lrs or [1e-3, 1e-4]  
        _batches = batch_sizes or [256] 
        _weight_decays = weight_decays or [0,1e-5] 
        
        _seq_lens = seq_lens or [24, 48] 
        _hidden = lstm_hidden_dims or [64, 128]  
        _nlayers = lstm_num_layers or [1, 2]  
        
        _nus = ocsvm_nus or [0.01, 0.05, 0.1] 
        _gammas = ocsvm_gammas or ["scale", "auto"]  
        _iforest_conts = iforest_contaminations or [0.01, 0.05] 
        _iforest_trees = iforest_n_estimators or [100, 300] 
        _debounces = debounce_consecutives or [1, 4, 6] 
        
        _epochs, _patience = epochs, patience

    elif mode == "extensive":
        _models = models or ["dense", "lstm", "ocsvm", "iforest"]
        _presets = presets or available_presets
        _thresholds = thresholds or [95.0, 97.5, 99.0, 99.5]
        _val_starts = val_start_dates or default_val_starts
        
        _layers = dense_layers or [
            None, (64, 32), (128, 64), (256, 128),
            (64, 32, 16), (128, 64, 32), (256, 128, 64),
        ]
        _lrs = dense_lrs or [1e-3, 5e-4, 1e-4, 5e-5]
        _batches = batch_sizes or [128, 256, 512]
        _weight_decays = weight_decays or [0, 1e-5, 1e-4]
        
        _seq_lens = seq_lens or [12, 24, 36, 48, 72]
        _hidden = lstm_hidden_dims or [32, 64, 96, 128]
        _nlayers = lstm_num_layers or [1, 2, 3]
        
        _nus = ocsvm_nus or [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
        _gammas = ocsvm_gammas or ["scale", "auto", "0.001", "0.01", "0.1"]
        _iforest_conts = iforest_contaminations or [0.001, 0.005, 0.01, 0.05, 0.1]
        _iforest_trees = iforest_n_estimators or [100, 200, 300, 500]
        _debounces = debounce_consecutives or [1, 2, 4, 6, 12]
        
        _epochs, _patience = epochs * 2, patience * 2

    else:  # full
        _models = models or ["dense", "lstm", "ocsvm", "iforest"]
        _presets = presets or available_presets
        _thresholds = thresholds or [90.0, 95.0, 97.5, 99.0]
        _val_starts = val_start_dates or default_val_starts
        
        _layers = dense_layers or [
            None, (64, 32, 16), (128, 64, 32),
            (256, 128, 64), (128, 64, 32, 16),
        ]
        _lrs = dense_lrs or [1e-3, 5e-4, 1e-4]
        _batches = batch_sizes or [128, 256, 512]
        _weight_decays = weight_decays or [0, 1e-5]
        
        _seq_lens = seq_lens or [12, 24, 48]
        _hidden = lstm_hidden_dims or [32, 64, 128]
        _nlayers = lstm_num_layers or [1, 2, 3]
        
        _nus = ocsvm_nus or [0.005, 0.01, 0.05, 0.1]
        _gammas = ocsvm_gammas or ["scale", "auto", 0.01]
        _iforest_conts = iforest_contaminations or [0.005, 0.01, 0.05]
        _iforest_trees = iforest_n_estimators or [100, 200, 500]
        _debounces = debounce_consecutives or [1, 4, 6]
        
        _epochs, _patience = epochs, patience

    trials: list[TrialConfig] = []

    for val_start, preset, model, threshold, debounce in product(
        _val_starts, _presets, _models, _thresholds, _debounces
    ):
        if model == "dense":
            for lr, bs, layers, wd in product(_lrs, _batches, _layers, _weight_decays):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold, debounce_consecutive=debounce,
                    learning_rate=lr, batch_size=bs, dense_layers=layers,
                    weight_decay=wd,
                    epochs=_epochs, patience=_patience,
                ))

        elif model == "lstm":
            for sl, hd, nl in product(_seq_lens, _hidden, _nlayers):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold, debounce_consecutive=debounce,
                    seq_len=sl, lstm_hidden_dim=hd, lstm_num_layers=nl,
                    epochs=_epochs, patience=_patience,
                ))

        elif model == "ocsvm":
            for nu, gamma in product(_nus, _gammas):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold, debounce_consecutive=debounce,
                    ocsvm_nu=nu, ocsvm_gamma=gamma,
                ))

        elif model == "iforest":
            for cont, trees in product(_iforest_conts, _iforest_trees):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold, debounce_consecutive=debounce,
                    iforest_contamination=cont, iforest_n_estimators=trees,
                ))

    return trials


# ════════════════════════════════════════════════════════════════
# Trial Runner
# ════════════════════════════════════════════════════════════════

def run_trial(
    trial: TrialConfig,
    equipment_id: str,
    df_pre: pd.DataFrame,
    device: str,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
    logger=None,
    trial_idx: int = 0,
) -> dict[str, Any] | None:
    """Executa um trial completo."""
    config = EQUIPMENT_CONFIGS[equipment_id]
    min_rows = max(50, trial.seq_len + 1 if trial.model == "lstm" else 50)

    splits = temporal_split(
        df_pre,
        val_start_date=trial.val_start,
        val_end_date=getattr(config, "val_end_date", None),
    )

    if len(splits["train"]) < min_rows or len(splits["val"]) < min_rows:
        return None

    steps = get_preprocessing_steps(equipment_id, preset=trial.preset)
    train_df, artifacts, _ = run_preprocessing(
        splits["train"], steps, return_artifacts=True, return_report=True
    )
    val_df, _, _ = run_preprocessing(
        splits["val"], steps, fitted_artifacts=artifacts,
        return_artifacts=True, return_report=True
    )
    full_raw = pd.concat([splits["train"], splits["val"], splits["test"]]).sort_index()
    full_df, _, _ = run_preprocessing(
        full_raw, steps, fitted_artifacts=artifacts,
        return_artifacts=True, return_report=True
    )

    if len(train_df) < min_rows or len(val_df) < min_rows:
        return None

    model, val_loss = train_model(
        trial.model, train_df, val_df, device,
        dense_layers=list(trial.dense_layers) if trial.dense_layers else None,
        seq_len=trial.seq_len,
        lstm_hidden_dim=trial.lstm_hidden_dim,
        lstm_num_layers=trial.lstm_num_layers,
        batch_size=trial.batch_size,
        epochs=trial.epochs,
        patience=trial.patience,
        learning_rate=trial.learning_rate,
        weight_decay=trial.weight_decay,
        ocsvm_nu=trial.ocsvm_nu,
        ocsvm_gamma=trial.ocsvm_gamma,
        iforest_contamination=trial.iforest_contamination,
        iforest_n_estimators=trial.iforest_n_estimators,
        logger=logger,
    )

    scores_df, threshold, train_errors = score_full(
        model, trial.model, train_df, full_df,
        trial.threshold_percentile, device,
        seq_len=trial.seq_len, batch_size=trial.batch_size,
    )

    n_anomalies = int(scores_df["is_anomaly"].sum())
    anomaly_rate = float(scores_df["is_anomaly"].mean())

    # MÉTRICAS balanceadas com penalização de FP
    failure_date = getattr(config, "failure_date", None)

    if failure_date is not None:
        metrics = compute_balanced_score(
            scores_df,
            failure_date=failure_date,
            prefailure_days=prefailure_days,
            normal_end_days=normal_end_days,
            false_positive_penalty=2.0,
            min_prefailure_rate=0.3,
            debounce_consecutive=trial.debounce_consecutive,
        )
        composite_score = float(metrics["composite_score"])
    else:
        metrics = {
            "composite_score": 1.0 - anomaly_rate,
            "balanced_score": -anomaly_rate,
            "discrimination_ratio": 0.0,
            "prefailure_alert_rate": 0.0,
            "normal_alert_rate": anomaly_rate,
            "n_prefailure_alerts": 0,
            "n_normal_alerts": n_anomalies,
            "n_prefailure_samples": 0,
            "n_normal_samples": len(scores_df),
            "false_positive_penalty_used": 2.0,
            "min_prefailure_rate_used": 0.3,
        }
        composite_score = 1.0 - anomaly_rate

    if logger:
        series = f"automl/{trial.model}"
        logger.report_scalar(series, "val_loss", val_loss, trial_idx)
        logger.report_scalar(series, "composite_score", composite_score, trial_idx)
        logger.report_scalar(series, "normal_alert_rate", float(metrics["normal_alert_rate"]), trial_idx)

    row = trial.to_dict()
    row.update({
        "trial_label": trial.label(),
        "val_loss": val_loss,
        "threshold": threshold,
        "train_score_mean": float(train_errors.mean()),
        "train_score_std": float(train_errors.std()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "scored_samples": len(scores_df),
        "composite_score": composite_score,
        "pct_anomalies": anomaly_rate,
        "_model": model,
        "_scores_df": scores_df,
        "_artifacts": artifacts,
    })
    row.update(metrics)

    return row


# ════════════════════════════════════════════════════════════════
# Ranking
# ════════════════════════════════════════════════════════════════

def rank_results(rows: list[dict[str, Any]], max_fp_rate: float | None = None) -> pd.DataFrame:
    """Ordena trials com suporte a constraint de FP."""
    df = pd.DataFrame(rows).reset_index(drop=True)

    if max_fp_rate is not None and max_fp_rate > 0:
        viable = df[df["normal_alert_rate"] <= max_fp_rate]

        if len(viable) == 0:
            print(f"\nAVISO: Nenhum modelo atingiu max_fp_rate={max_fp_rate:.2%}")
            print("Retornando ordenado por menor taxa de FP...\n")
            return df.sort_values(
                ["normal_alert_rate", "prefailure_alert_rate"],
                ascending=[True, False],
            ).reset_index(drop=True)

        return viable.sort_values(
            ["prefailure_alert_rate", "composite_score", "normal_alert_rate"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    return df.sort_values(
        ["composite_score", "prefailure_alert_rate", "normal_alert_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# Formatting & Display
# ════════════════════════════════════════════════════════════════

def print_trial_header(trial_num: int, total_trials: int, trial: TrialConfig) -> None:
    print(f"\n[{trial_num:03d}/{total_trials:03d}] {trial.label()}")
    print("─" * 70)


def print_trial_result(result: dict[str, Any]) -> None:
    indent = "  "
    if result.get("val_loss", 0.0) > 0.0:
        print(f"{indent}Val Loss: {result['val_loss']:.5f}")
    print(f"{indent}Threshold: {result['threshold']:.5f}")
    print(f"{indent}Anomalias: {result['n_anomalies']}/{result['scored_samples']} ({result['anomaly_rate']:.2%})")
    
    pre = result.get("prefailure_alert_rate", 0.0)
    norm = result.get("normal_alert_rate", 0.0)
    print(f"{indent}Pre-falha: {pre:.2%}  |  Normal: {norm:.2%}")
    print(f"{indent}Score: {result['composite_score']:.5f}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main(
    equipment_id: str,
    mode: ModeType = "full",
    remote: bool = False,
    local_data: bool = False,
    queue: str = "default",
    upload_to_clearml: bool = True,
    local_artifacts_dir: str = "artifacts_local",
    models: list[str] | None = None,
    epochs: int = 100,
    patience: int = 15,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
    max_fp_rate: float | None = None,
    clearml_project=None,
) -> None:
    """Pipeline principal do AutoML."""
    start_time = time.time()
    config = EQUIPMENT_CONFIGS[equipment_id]

    print("=" * 70)
    print(f"{'AUTOML - DETECÇÃO DE ANOMALIAS (OTIMIZADO)':^70}")
    print("=" * 70)
    print(f"Equipamento: {equipment_id}")
    print(f"Modo: {mode}")
    print(f"Max FP Rate: {max_fp_rate:.2%}" if max_fp_rate else "Max FP Rate: ∞ (sem constraint)")
    print("=" * 70 + "\n")

    Task.add_requirements("setuptools>=65.0")  # FIX para OpenSSL 3.0
    Task.add_requirements("gitpython>=3.1.40")  # FIX para clone
    Task.add_requirements("pyarrow")
    Task.add_requirements("torch", package_version="")

    task = Task.init(
        project_name=clearml_project or "Transpetro",
        task_name=f"automl_{equipment_id}_{mode}",
        output_uri=True,
    )

    task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")

    if remote:
        print(f"Executando remotamente na fila: {queue}")
        task.execute_remotely(queue_name=queue)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = task.get_logger()
    print(f"Device: {device}\n")

    # LOAD DATA
    print("Carregando dados...")
    df_raw = load_equipment_data(equipment_id, from_clearml=not local_data)
    print(f"  Shape (RAW):        {df_raw.shape}")
    df_pre, _, _ = run_preprocessing(df_raw, config.pre_split_steps)
    print(f"  Shape: {df_pre.shape}\n")

    # BUILD TRIALS
    print("Gerando grid de trials...")
    trials = build_trials(equipment_id, mode=mode, models=models, epochs=epochs, patience=patience)
    print(f"  Total: {len(trials)} trials\n")

    # EXECUTE
    print("=" * 70)
    print(f"{'EXECUÇÃO':^70}")
    print("=" * 70)

    rows: list[dict] = []
    best_row = None
    n_skipped = 0
    n_failed = 0

    for i, trial in enumerate(trials, 1):
        print_trial_header(i, len(trials), trial)

        try:
            row = run_trial(
                trial, equipment_id, df_pre, device,
                prefailure_days, normal_end_days, logger, i
            )

            if row is None:
                print("[SKIP] Dados insuficientes")
                n_skipped += 1
                continue

            print_trial_result(row)
            
            # Atualiza melhor (respeitando constraint de FP)
            should_update = False
            if best_row is None:
                should_update = max_fp_rate is None or row["normal_alert_rate"] <= max_fp_rate
            else:
                if max_fp_rate is None:
                    should_update = row["composite_score"] > best_row["composite_score"]
                else:
                    row_fp_ok = row["normal_alert_rate"] <= max_fp_rate
                    best_fp_ok = best_row["normal_alert_rate"] <= max_fp_rate
                    
                    if row_fp_ok and best_fp_ok:
                        should_update = row["prefailure_alert_rate"] > best_row["prefailure_alert_rate"]
                    elif row_fp_ok and not best_fp_ok:
                        should_update = True

            if should_update:
                if best_row is not None:
                    best_row.pop("_model", None)
                    best_row.pop("_scores_df", None)
                    best_row.pop("_artifacts", None)
                best_row = row
            else:
                row.pop("_model", None)
                row.pop("_scores_df", None)
                row.pop("_artifacts", None)

            rows.append({k: v for k, v in row.items() if not k.startswith("_")})

        except Exception as e:
            print(f"  [ERRO] {e}")
            n_failed += 1

    if best_row is None:
        raise RuntimeError("Todos os trials falharam.")

    # RESULTS
    print("\n" + "=" * 70)
    print(f"{'RESULTADOS':^70}")
    print("=" * 70)

    ranking = rank_results(rows, max_fp_rate=max_fp_rate)
    print("\nTOP 10:\n")
    print(ranking[[c for c in ["composite_score", "prefailure_alert_rate", "normal_alert_rate", "model"] 
                   if c in ranking.columns]].head(10).to_string())

    if local_artifacts_dir:
        output_dir = Path(local_artifacts_dir) / f"{equipment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSalvando artifacts em: {output_dir}")
        
        ranking.to_parquet(output_dir / "ranking.parquet")
        print(f"✓ Ranking salvo")

        best_trial_dict = {k: v for k, v in best_row.items() if not k.startswith("_")}
        with open(output_dir / "best_trial.json", "w") as f:
            json.dump(best_trial_dict, f, indent=2, default=str)
        print(f"✓ Best trial salvo")
        
        if "_model" in best_row and best_row["_model"] is not None:
            model = best_row["_model"]
            model_type = best_row["model"]
            
            if model_type in ["ocsvm", "iforest"]:
                model_path = output_dir / f"best_model_{equipment_id}.pkl"
                with model_path.open("wb") as f:
                    pickle.dump(model, f)
            else:
                model_path = output_dir / f"best_model_{equipment_id}.pt"
                torch.save(model.state_dict(), model_path)
            
            print(f"✓ Modelo salvo")
        
        if "_scores_df" in best_row and best_row["_scores_df"] is not None:
            scores_path = output_dir / "best_scores.parquet"
            best_row["_scores_df"].to_parquet(scores_path)
            print(f"✓ Scores salvos")
        
        if upload_to_clearml and task is not None:
            print(f"\n Upload ao ClearML...")
            task.upload_artifact("automl_ranking", artifact_object=ranking)
            if "_scores_df" in best_row:
                task.upload_artifact("best_scores", artifact_object=best_row["_scores_df"])

            task.upload_artifact("best_trial", artifact_object=best_trial_dict)
            print(f"✓ Upload completo")

    duration = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Duração: {duration / 3600:.1f}h | Trials: {len(rows)}/{len(trials)} | "
          f"Falhas: {n_failed} | Pulados: {n_skipped}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML para detecção de anomalias (OTIMIZADO)")
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    parser.add_argument("--mode", choices=["quick", "full", "balanced", "extensive"], default="full")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--queue", default="default")
    parser.add_argument("--local-data", action="store_true")
    parser.add_argument("--no-clearml-upload", action="store_true")
    parser.add_argument("--local-artifacts-dir", default="artifacts_local")
    parser.add_argument("--models", nargs="+", default=None, choices=["dense", "lstm", "ocsvm", "iforest"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--prefailure-days", type=int, default=30)
    parser.add_argument("--normal-end-days", type=int, default=60)
    parser.add_argument("--clearml-project", default="Transpetro")
    parser.add_argument(
        "--max-fp-rate", type=float, default=0.0,
        help="Taxa máxima de FP permitida (0-1). Use 0 para desabilitar constraint."
    )

    args = parser.parse_args()

    main(
        equipment_id=args.equipment,
        mode=args.mode,
        remote=args.remote,
        local_data=args.local_data,
        queue=args.queue,
        upload_to_clearml=not args.no_clearml_upload,
        local_artifacts_dir=args.local_artifacts_dir,
        models=args.models,
        epochs=args.epochs,
        patience=args.patience,
        prefailure_days=args.prefailure_days,
        normal_end_days=args.normal_end_days,
        max_fp_rate=args.max_fp_rate if args.max_fp_rate > 0 else None,
        clearml_project=args.clearml_project,
    )