"""
AutoML para detecção de anomalias: grid search sobre modelos, presets e hiperparâmetros.

Funções genéricas reutilizáveis:
  train_model()   — treina qualquer modelo suportado (dense, lstm, ocsvm)
  score_full()    — computa scores e threshold sobre um DataFrame completo
  build_trials()  — gera grade de TrialConfig para um equipamento
  run_trial()     — executa um trial e retorna métricas
  rank_results()  — ordena resultados por composite_score
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch

from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.data.splitting import temporal_split
from transpetro_modelos.models.autoencoder import DenseAutoencoder, LSTMAutoencoder, VAE
from transpetro_modelos.training.evaluate import (
    apply_debounce,
    cusum_anomaly_score,
    compute_isolation_forest_errors,
    compute_lof_errors,
    compute_ocsvm_errors,
    compute_reconstruction_errors,
    compute_reconstruction_errors_sequence,
    compute_vae_errors,
    determine_threshold,
    failure_detection_metrics,
    fit_isolation_forest,
    fit_lof,
    fit_ocsvm,
    score_isolation_forest_set,
    score_lof_set,
    score_ocsvm_set,
    score_test_set,
    score_test_set_sequence,
    score_vae_set,
)
from transpetro_modelos.training.train import (
    make_dataloader,
    make_sequence_dataloader,
    train_autoencoder,
    train_vae,
)


# ── Generic helpers ────────────────────────────────────────────────────────────


def build_model(
    model_type: str,
    n_features: int,
    *,
    dense_layers: list[int] | None = None,
    seq_len: int = 24,
    lstm_hidden_dim: int = 64,
    lstm_num_layers: int = 2,
    latent_dim: int = 8,
):
    """Instancia a arquitetura (PyTorch) SEM treinar — usada para recarregar pesos
    salvos (state_dict) no deploy. Só faz sentido para modelos PyTorch (dense/lstm/vae);
    modelos sklearn (ocsvm/iforest/lof) são serializados inteiros via pickle."""
    if model_type == "vae":
        return VAE(
            input_dim=n_features,
            encoding_layers=list(dense_layers) if dense_layers else None,
            latent_dim=latent_dim,
        )
    if model_type == "lstm":
        return LSTMAutoencoder(
            input_dim=n_features,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            seq_len=seq_len,
        )
    if model_type == "dense":
        return DenseAutoencoder(
            input_dim=n_features,
            encoding_layers=list(dense_layers) if dense_layers else None,
        )
    raise ValueError(f"build_model não suporta model_type={model_type!r} (use pickle p/ sklearn)")

def train_model(
    model_type: str,
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
    ocsvm_gamma: str = "scale",
    latent_dim: int = 8,
    vae_beta: float = 1.0,
    if_n_estimators: int = 100,
    if_contamination: float = 0.05,
    lof_n_neighbors: int = 20,
    lof_contamination: float = 0.05,
):
    """Treina e retorna o modelo. Suporta dense, lstm, ocsvm, vae, isolation_forest e lof."""
    if model_type == "ocsvm":
        return fit_ocsvm(train_df, nu=ocsvm_nu, gamma=ocsvm_gamma)

    if model_type == "isolation_forest":
        return fit_isolation_forest(train_df, n_estimators=if_n_estimators, contamination=if_contamination)

    if model_type == "lof":
        return fit_lof(train_df, n_neighbors=lof_n_neighbors, contamination=lof_contamination)

    n_features = train_df.shape[1]

    if model_type == "vae":
        model = VAE(
            input_dim=n_features,
            encoding_layers=list(dense_layers) if dense_layers else None,
            latent_dim=latent_dim,
        ).to(device)
        train_loader = make_dataloader(train_df, batch_size=batch_size, shuffle=True, device=device)
        val_loader = make_dataloader(val_df, batch_size=batch_size, shuffle=False, device=device)
        return train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            beta=vae_beta,
            logger=logger,
        )

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
        train_loader = make_dataloader(train_df, batch_size=batch_size, shuffle=True, device=device)
        val_loader = make_dataloader(val_df, batch_size=batch_size, shuffle=False, device=device)

    return train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        logger=logger,
    )


def score_full(
    model,
    model_type: str,
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    threshold_percentile: float,
    device: str,
    *,
    seq_len: int = 24,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """
    Determina o threshold a partir do train_df e aplica sobre o full_df.
    Retorna (scores_df, threshold, train_errors).
    """
    if model_type == "ocsvm":
        train_errors = compute_ocsvm_errors(model, train_df)
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        return score_ocsvm_set(model, full_df, threshold), threshold, train_errors

    if model_type == "isolation_forest":
        train_errors = compute_isolation_forest_errors(model, train_df)
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        return score_isolation_forest_set(model, full_df, threshold), threshold, train_errors

    if model_type == "lof":
        train_errors = compute_lof_errors(model, train_df)
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        return score_lof_set(model, full_df, threshold), threshold, train_errors

    if model_type == "vae":
        train_errors = compute_vae_errors(model, train_df, device=device, batch_size=batch_size)
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        scores = score_vae_set(model, full_df, threshold, device=device, batch_size=batch_size)
        return scores, threshold, train_errors

    if model_type == "lstm":
        train_errors = compute_reconstruction_errors_sequence(
            model, train_df, seq_len=seq_len, batch_size=batch_size, device=device
        )
        threshold = determine_threshold(train_errors, percentile=threshold_percentile)
        scores = score_test_set_sequence(
            model, full_df, seq_len=seq_len, threshold=threshold,
            batch_size=batch_size, device=device,
        )
        return scores, threshold, train_errors

    train_errors = compute_reconstruction_errors(model, train_df, batch_size=batch_size, device=device)
    threshold = determine_threshold(train_errors, percentile=threshold_percentile)
    scores = score_test_set(model, full_df, threshold=threshold, batch_size=batch_size, device=device)
    return scores, threshold, train_errors


# ── TrialConfig ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrialConfig:
    val_start: datetime | None  # None = usa val_fraction=0.2 do temporal_split
    preset: str
    model: str
    threshold_percentile: float
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10
    dense_layers: tuple[int, ...] | None = None
    seq_len: int = 24
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    ocsvm_nu: float = 0.05
    ocsvm_gamma: str = "scale"
    latent_dim: int = 8
    vae_beta: float = 1.0
    if_n_estimators: int = 100
    if_contamination: float = 0.05
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.05
    debounce_consecutive: int = 1  # 1 = sem debounce; >1 = N pontos consecutivos necessários
    alarm_policy: str = "threshold"  # "threshold" (percentil, default) ou "cusum" (deriva sustentada)
    cusum_k: float = 0.5  # folga do CUSUM em unidades de std (só usado se alarm_policy="cusum")
    cusum_h: float = 5.0  # limiar de decisão do CUSUM em unidades de std

    def label(self) -> str:
        vs = self.val_start.strftime("%Y-%m-%d") if self.val_start else "auto"
        parts = [vs, self.preset, self.model, f"p{self.threshold_percentile:g}"]
        if self.model == "dense":
            parts.extend([f"lr{self.learning_rate:g}", f"b{self.batch_size}"])
            if self.dense_layers:
                parts.append("layers" + "-".join(str(v) for v in self.dense_layers))
        elif self.model == "lstm":
            parts.extend([
                f"seq{self.seq_len}", f"h{self.lstm_hidden_dim}", f"l{self.lstm_num_layers}",
            ])
        elif self.model == "vae":
            parts.extend([f"lr{self.learning_rate:g}", f"b{self.batch_size}", f"latent{self.latent_dim}"])
            if self.dense_layers:
                parts.append("layers" + "-".join(str(v) for v in self.dense_layers))
        elif self.model == "isolation_forest":
            parts.extend([f"nest{self.if_n_estimators}", f"cont{self.if_contamination:g}"])
        elif self.model == "lof":
            parts.extend([f"k{self.lof_n_neighbors}", f"cont{self.lof_contamination:g}"])
        else:  # ocsvm
            parts.extend([f"nu{self.ocsvm_nu:g}", f"gamma{self.ocsvm_gamma}"])
        if self.debounce_consecutive > 1:
            parts.append(f"deb{self.debounce_consecutive}")
        if self.alarm_policy != "threshold":
            parts.append(f"{self.alarm_policy}-k{self.cusum_k:g}-h{self.cusum_h:g}")
        return "__".join(parts)


# ── Grid builder ───────────────────────────────────────────────────────────────

def build_trials(
    equipment_id: str,
    *,
    presets: list[str] | None = None,
    models: list[str] | None = None,
    thresholds: list[float] | None = None,
    val_start_dates: list[datetime | None] | None = None,
    dense_layers: list[tuple[int, ...] | None] | None = None,
    dense_lrs: list[float] | None = None,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
    lstm_hidden_dims: list[int] | None = None,
    lstm_layers: list[int] | None = None,
    ocsvm_nus: list[float] | None = None,
    ocsvm_gammas: list[str] | None = None,
    latent_dims: list[int] | None = None,
    if_n_estimators_list: list[int] | None = None,
    if_contamination_list: list[float] | None = None,
    lof_n_neighbors_list: list[int] | None = None,
    lof_contamination_list: list[float] | None = None,
    debounce_consecutives: list[int] | None = None,
    alarm_policies: list[str] | None = None,
    epochs: int = 100,
    patience: int = 10,
    quick: bool = False,
    mode: str = "full",
) -> list[TrialConfig]:
    """
    Gera a grade de trials para um equipamento.

    mode:
      quick     — smoke test (5-30 min): poucos modelos, epochs=20
      full      — busca balanceada (~1 dia): todos os modelos, grade razoável
      extensive — exploração profunda (~2 dias): grade densa em todos os eixos

    Se `presets` não fornecido, auto-detecta via config.preprocess_presets.
    Se `val_start_dates` não fornecido, usa config.val_start_date ou None.
    """
    config = EQUIPMENT_CONFIGS[equipment_id]
    available_presets = list(config.preprocess_presets.keys()) if config.preprocess_presets else ["baseline"]
    default_val_starts: list[datetime | None] = (
        [config.val_start_date] if config.val_start_date else [None]
    )

    # --mode quick sobrescreve o flag legado --quick
    if quick:
        mode = "quick"

    if mode == "quick":
        _presets       = presets       or available_presets[:1]
        _models        = models        or ["dense", "ocsvm", "isolation_forest", "lof"]
        _thresholds    = thresholds    or [95.0, 99.0]
        _val_starts    = val_start_dates or default_val_starts
        _dense_layers  = dense_layers  or [None]
        _dense_lrs     = dense_lrs     or [1e-3]
        _batch_sizes   = batch_sizes   or [256]
        _seq_lens      = seq_lens      or [24]
        _lstm_hidden_dims = lstm_hidden_dims or [64]
        _lstm_layers   = lstm_layers   or [2]
        _ocsvm_nus     = ocsvm_nus     or [0.05]
        _ocsvm_gammas  = ocsvm_gammas  or ["scale"]
        _latent_dims   = latent_dims   or [8]
        _if_n_estimators   = if_n_estimators_list   or [100]
        _if_contaminations = if_contamination_list  or [0.05]
        _lof_n_neighbors   = lof_n_neighbors_list   or [20]
        _lof_contaminations = lof_contamination_list or [0.05]
        _debounces     = debounce_consecutives or [1]
        _epochs, _patience = 20, 5

    elif mode == "extensive":
        _presets       = presets       or available_presets
        _models        = models        or ["dense", "lstm", "ocsvm", "vae", "isolation_forest", "lof"]
        _thresholds    = thresholds    or [90.0, 92.5, 95.0, 97.5, 99.0]
        _val_starts    = val_start_dates or default_val_starts
        _dense_layers  = dense_layers  or [None, (32,), (64, 32), (64, 32, 16), (128, 64, 32), (256, 128, 64), (128, 64, 32, 16)]
        _dense_lrs     = dense_lrs     or [1e-3, 5e-4, 1e-4, 5e-5]
        _batch_sizes   = batch_sizes   or [64, 128, 256, 512]
        _seq_lens      = seq_lens      or [12, 24, 48, 72, 96]
        _lstm_hidden_dims = lstm_hidden_dims or [32, 64, 128, 256]
        _lstm_layers   = lstm_layers   or [1, 2, 3]
        _ocsvm_nus     = ocsvm_nus     or [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
        _ocsvm_gammas  = ocsvm_gammas  or ["scale", "auto", "0.001", "0.01", "0.1"]
        _latent_dims   = latent_dims   or [4, 8, 16, 32]
        _if_n_estimators   = if_n_estimators_list   or [100, 200, 300, 500]
        _if_contaminations = if_contamination_list  or [0.001, 0.005, 0.01, 0.05, 0.1]
        _lof_n_neighbors   = lof_n_neighbors_list   or [10, 20, 50, 100]
        _lof_contaminations = lof_contamination_list or [0.001, 0.005, 0.01, 0.05, 0.1]
        _debounces     = debounce_consecutives or [1, 2, 4, 6, 12]
        _epochs, _patience = epochs * 2, patience * 2

    if mode == "balanced":
        _presets       = presets       or available_presets[:2]
        _models        = models        or ["dense", "lstm", "ocsvm", "vae", "isolation_forest", "lof"]
        _thresholds    = thresholds    or [95.0, 97.5, 99.0]
        _val_starts    = val_start_dates or default_val_starts
        _dense_layers  = dense_layers  or [None, (64, 32, 16), (128, 64, 32), (256, 128, 64)]
        _dense_lrs     = dense_lrs     or [1e-3, 1e-4]
        _batch_sizes   = batch_sizes   or [128, 256]
        _seq_lens      = seq_lens      or [24, 48]
        _lstm_hidden_dims = lstm_hidden_dims or [64, 128]
        _lstm_layers   = lstm_layers   or [1, 2]
        _ocsvm_nus     = ocsvm_nus     or [0.001, 0.005, 0.01, 0.05, 0.1]
        _ocsvm_gammas  = ocsvm_gammas  or ["scale", "auto", "0.001", "0.01", "0.1"]
        _latent_dims   = latent_dims   or [8, 16, 32]
        _if_n_estimators   = if_n_estimators_list   or [100, 200, 300]
        _if_contaminations = if_contamination_list  or [0.001, 0.005, 0.01, 0.05, 0.1]
        _lof_n_neighbors   = lof_n_neighbors_list   or [10, 20, 50]
        _lof_contaminations = lof_contamination_list or [0.001, 0.005, 0.01, 0.05, 0.1]
        _debounces     = debounce_consecutives or [1, 4, 6]
        _epochs, _patience = epochs, patience

    else:  # full (default, ~1 dia)
        _presets       = presets       or available_presets
        _models        = models        or ["dense", "lstm", "ocsvm", "vae", "isolation_forest", "lof"]
        _thresholds    = thresholds    or [90.0, 95.0, 97.5, 99.0]
        _val_starts    = val_start_dates or default_val_starts
        _dense_layers  = dense_layers  or [None, (64, 32, 16), (128, 64, 32), (256, 128, 64), (128, 64, 32, 16), (64, 32)]
        _dense_lrs     = dense_lrs     or [1e-3, 5e-4, 1e-4, 5e-5]
        _batch_sizes   = batch_sizes   or [128, 256, 512]
        _seq_lens      = seq_lens      or [12, 24, 48, 72]
        _lstm_hidden_dims = lstm_hidden_dims or [32, 64, 128, 256]
        _lstm_layers   = lstm_layers   or [1, 2, 3]
        _ocsvm_nus     = ocsvm_nus     or [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
        _ocsvm_gammas  = ocsvm_gammas  or ["scale", "auto", "0.001", "0.01", "0.1"]
        _latent_dims   = latent_dims   or [8, 16, 32]
        _if_n_estimators   = if_n_estimators_list   or [100, 200, 300, 500]
        _if_contaminations = if_contamination_list  or [0.001, 0.005, 0.01, 0.05, 0.1]
        _lof_n_neighbors   = lof_n_neighbors_list   or [10, 20, 50, 100]
        _lof_contaminations = lof_contamination_list or [0.001, 0.005, 0.01, 0.05, 0.1]
        _debounces     = debounce_consecutives or [1, 4, 6]
        _epochs, _patience = epochs, patience

    trials: list[TrialConfig] = []
    for val_start, preset, model, threshold, deb in product(_val_starts, _presets, _models, _thresholds, _debounces):
        if model == "dense":
            for lr, bs, layers in product(_dense_lrs, _batch_sizes, _dense_layers):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold,
                    learning_rate=lr, batch_size=bs, dense_layers=layers,
                    epochs=_epochs, patience=_patience,
                    debounce_consecutive=deb,
                ))
        elif model == "lstm":
            for sl, hd, nl in product(_seq_lens, _lstm_hidden_dims, _lstm_layers):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold,
                    seq_len=sl, lstm_hidden_dim=hd, lstm_num_layers=nl,
                    epochs=_epochs, patience=_patience,
                    debounce_consecutive=deb,
                ))
        elif model == "ocsvm":
            for nu, gamma in product(_ocsvm_nus, _ocsvm_gammas):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold,
                    ocsvm_nu=nu, ocsvm_gamma=gamma,
                    debounce_consecutive=deb,
                ))
        elif model == "vae":
            for lr, bs, layers, ldim in product(_dense_lrs, _batch_sizes, _dense_layers, _latent_dims):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold,
                    learning_rate=lr, batch_size=bs, dense_layers=layers,
                    latent_dim=ldim,
                    epochs=_epochs, patience=_patience,
                    debounce_consecutive=deb,
                ))
        elif model == "isolation_forest":
            for n_est, cont in product(_if_n_estimators, _if_contaminations):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold,
                    if_n_estimators=n_est,
                    if_contamination=cont,
                    debounce_consecutive=deb,
                ))
        elif model == "lof":
            for k, cont in product(_lof_n_neighbors, _lof_contaminations):
                trials.append(TrialConfig(
                    val_start=val_start, preset=preset, model=model,
                    threshold_percentile=threshold,
                    lof_n_neighbors=k,
                    lof_contamination=cont,
                    debounce_consecutive=deb,
                ))
        else:
            raise ValueError(f"Modelo desconhecido: {model}")

    # Política de alarme (opt-in). Default ["threshold"] → nenhuma expansão (idêntico ao original).
    # "cusum" gera variantes do mesmo trial com alarme por deriva. OBS: como o CUSUM ignora o
    # threshold_percentile, restrinja --thresholds (ex.: um só) ao rodar cusum para não inflar a grade.
    _policies = alarm_policies or ["threshold"]
    if _policies != ["threshold"]:
        expanded: list[TrialConfig] = []
        for t in trials:
            for pol in _policies:
                expanded.append(t if pol == "threshold" else replace(t, alarm_policy=pol))
        trials = expanded

    return trials


# ── Trial runner ───────────────────────────────────────────────────────────────

def run_trial(
    trial: TrialConfig,
    equipment_id: str,
    df_pre: pd.DataFrame,
    device: str,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
) -> dict[str, Any] | None:
    """
    Executa um único trial: split → preprocessing → treino → score → métricas.
    Retorna None se os dados forem insuficientes.
    """
    config = EQUIPMENT_CONFIGS[equipment_id]
    min_rows = max(50, trial.seq_len + 1 if trial.model == "lstm" else 50)

    splits = temporal_split(
        df_pre,
        failure_date=config.failure_date,
        exclusion_days=config.exclusion_days_before,
        val_start_date=trial.val_start,
        val_end_date=config.val_end_date,
    )
    if len(splits["train"]) < min_rows or len(splits["val"]) < min_rows:
        return None

    steps = get_preprocessing_steps(equipment_id, preset=trial.preset)
    train_df, artifacts, _ = run_preprocessing(splits["train"], steps, return_artifacts=True, return_report=True)
    val_df, _, _ = run_preprocessing(
        splits["val"], steps, fitted_artifacts=artifacts, return_artifacts=True, return_report=True
    )
    full_raw = pd.concat([splits["train"], splits["val"], splits["test"]]).sort_index()
    full_df, _, _ = run_preprocessing(
        full_raw, steps, fitted_artifacts=artifacts, return_artifacts=True, return_report=True
    )

    if len(train_df) < min_rows or len(val_df) < min_rows or len(full_df) < min_rows:
        return None

    model = train_model(
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
        latent_dim=trial.latent_dim,
        vae_beta=trial.vae_beta,
        if_n_estimators=trial.if_n_estimators,
        if_contamination=trial.if_contamination,
        lof_n_neighbors=trial.lof_n_neighbors,
        lof_contamination=trial.lof_contamination,
    )

    scores, threshold, train_errors = score_full(
        model, trial.model, train_df, full_df,
        trial.threshold_percentile, device,
        seq_len=trial.seq_len,
        batch_size=trial.batch_size,
    )

    # Política de alarme CUSUM (opt-in): recalcula is_anomaly por deriva sustentada,
    # calibrando mu/sigma nos erros de TREINO (sem vazamento). Default "threshold" inalterado.
    if trial.alarm_policy == "cusum":
        mu = float(np.mean(train_errors))
        sigma = float(np.std(train_errors))
        _, is_alarm = cusum_anomaly_score(
            scores["reconstruction_error"].values,
            k=trial.cusum_k, h=trial.cusum_h, mu=mu, sigma=sigma,
        )
        scores = scores.copy()
        scores["is_anomaly"] = is_alarm

    metrics = failure_detection_metrics(
        scores,
        config.failure_date,
        prefailure_days=prefailure_days,
        normal_end_days=normal_end_days,
        consecutive=trial.debounce_consecutive,
    )

    row = asdict(trial)
    row["val_start"] = trial.val_start.strftime("%Y-%m-%d") if trial.val_start else None
    row["dense_layers"] = "auto" if trial.dense_layers is None else ",".join(str(v) for v in trial.dense_layers)
    row.update({
        "threshold": threshold,
        "train_score_mean": float(train_errors.mean()),
        "train_score_std": float(train_errors.std()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "scored_samples": len(scores),
        "n_anomalies": int(scores["is_anomaly"].sum()),
        "pct_anomalies": float(scores["is_anomaly"].mean()),
    })
    row.update(metrics)

    # FP HELD-OUT: taxa de alarme medida na janela de VALIDAÇÃO (fora do treino), número honesto
    # de falso positivo (o normal_alert_rate padrão cai majoritariamente no train). Aditivo —
    # não altera o ranking; serve para auditar quão otimista é o FP in-sample.
    try:
        sc_db = apply_debounce(scores, consecutive=trial.debounce_consecutive)
        val_lo, val_hi = splits["val"].index.min(), splits["val"].index.max()
        vmask = (sc_db.index >= val_lo) & (sc_db.index <= val_hi)
        row["val_fp_rate_heldout"] = float(sc_db.loc[vmask, "is_anomaly"].mean()) if bool(vmask.any()) else None
        row["val_scored_samples"] = int(vmask.sum())
    except Exception:
        row["val_fp_rate_heldout"] = None
        row["val_scored_samples"] = 0

    return row


# ── Result ranking ─────────────────────────────────────────────────────────────

def rank_results(
    rows: list[dict[str, Any]],
    max_fp_rate: float | None = None,
    fp_column: str = "normal_alert_rate",
) -> pd.DataFrame:
    """
    Ordena trials priorizando baixos falsos positivos.

    Se max_fp_rate for fornecido, aplica uma constraint hard:
      - Apenas trials com FP <= max_fp_rate são considerados viáveis.
      - Dentro dos viáveis, ordena por prefailure_alert_rate (maximiza detecção).
      - Se nenhum trial satisfaz a constraint, retorna todos ordenados por
        FP crescente (o menos pior em FP primeiro).

    fp_column: coluna de FP usada na constraint/ordenação. Default "normal_alert_rate"
    (in-sample, comportamento original). Use "val_fp_rate_heldout" para o FP honesto
    medido na validação (auditoria A1); valores ausentes caem para o normal_alert_rate.

    Se max_fp_rate=None, usa o critério original: composite_score descendente.
    """
    df = pd.DataFrame(rows)

    if fp_column != "normal_alert_rate" and fp_column in df.columns:
        fp_sel = df[fp_column].fillna(df["normal_alert_rate"])
    else:
        fp_sel = df["normal_alert_rate"]

    if max_fp_rate is not None:
        df = df.assign(_fp_sel=fp_sel)
        viable = df[df["_fp_sel"] <= max_fp_rate]
        if len(viable) == 0:
            return (
                df.sort_values(
                    ["_fp_sel", "prefailure_alert_rate"],
                    ascending=[True, False],
                )
                .drop(columns=["_fp_sel"])
                .reset_index(drop=True)
            )
        return (
            viable.sort_values(
                ["prefailure_alert_rate", "_fp_sel"],
                ascending=[False, True],
            )
            .drop(columns=["_fp_sel"])
            .reset_index(drop=True)
        )

    return (
        df.sort_values(
            ["composite_score", "prefailure_alert_rate", "normal_alert_rate"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
