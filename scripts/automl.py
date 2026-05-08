"""
AutoML para Detecção de Anomalias
==================================
Modelos : Dense Autoencoder | LSTM Autoencoder | One-Class SVM
Seleção : Score composto (val_loss + anomaly_rate + threshold_ratio)
Busca   : Grid search declarativo via build_trials()
Modos   : Multivariado e Per-sensor (--per-sensor)

Uso rápido:
    python scripts/automl_anomaly_v3.py --equipment MEQ-01 --local-data --quick
Uso completo:
    python scripts/automl_anomaly_v3.py --equipment MEQ-01 --remote --queue gpu
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
    compute_ocsvm_errors,
    compute_reconstruction_errors,
    compute_reconstruction_errors_sequence,
    determine_threshold,
    failure_detection_metrics,
    fit_ocsvm,
    score_ocsvm_set,
    score_test_set,
    score_test_set_sequence,
)
from transpetro_modelos.training.train import (
    make_dataloader,
    make_sequence_dataloader,
    train_autoencoder,
)


# ── Type Aliases ───────────────────────────────────────────────────────────────

ModelType = Literal["dense", "lstm", "ocsvm"]
PresetName = str


# ── TrialConfig ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrialConfig:
    """
    Configuração imutável de um trial de AutoML.

    Suporta três tipos de modelo:
    - dense: Autoencoder denso com camadas customizáveis
    - lstm: Autoencoder sequencial para séries temporais
    - ocsvm: One-Class SVM para detecção de outliers
    """
    preset: str
    model: ModelType
    threshold_percentile: float

    # Splitting
    val_start: datetime | None = None  # None = usa val_fraction padrão do temporal_split

    # Dense hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10
    dense_layers: tuple[int, ...] | None = None

    # LSTM hyperparameters
    seq_len: int = 24
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2

    # OCSVM hyperparameters
    ocsvm_nu: float = 0.05
    ocsvm_gamma: str = "scale"

    def __post_init__(self):
        """Valida configuração após inicialização."""
        if self.model not in {"dense", "lstm", "ocsvm"}:
            raise ValueError(f"Modelo inválido: {self.model}. Use: dense, lstm ou ocsvm")

        if not 0 < self.threshold_percentile <= 100:
            raise ValueError(
                f"threshold_percentile deve estar entre 0 e 100, "
                f"recebido: {self.threshold_percentile}"
            )

        if self.model == "lstm" and self.seq_len < 1:
            raise ValueError(f"seq_len deve ser >= 1, recebido: {self.seq_len}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size deve ser >= 1, recebido: {self.batch_size}")

        if self.epochs < 1:
            raise ValueError(f"epochs deve ser >= 1, recebido: {self.epochs}")

    def label(self) -> str:
        """Gera identificador único e legível para o trial."""
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
        else:  # ocsvm
            parts.extend([f"nu{self.ocsvm_nu:g}", f"gamma_{self.ocsvm_gamma}"])

        return "__".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Converte para dict serializável (para logging/storage)."""
        d = asdict(self)
        d["val_start"] = self.val_start.strftime("%Y-%m-%d") if self.val_start else None
        d["dense_layers"] = (
            ",".join(str(v) for v in self.dense_layers)
            if self.dense_layers is not None
            else "auto"
        )
        return d


@dataclass
class PreprocessedData:
    """Container para dados pré-processados."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    full_df: pd.DataFrame
    artifacts: PreprocessingArtifacts


# ── Standalone model helpers ───────────────────────────────────────────────────

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
    ocsvm_gamma: str = "scale",
) -> tuple[Any, float]:
    """
    Treina qualquer modelo suportado e retorna (model, val_loss).

    Função standalone reutilizável fora do contexto de AutoML.
    Para OCSVM, val_loss = 0.0.
    """
    if model_type == "ocsvm":
        clf = fit_ocsvm(train_df, nu=ocsvm_nu, gamma=ocsvm_gamma)
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
    """
    Calcula threshold no train_df e aplica sobre full_df.

    Função standalone reutilizável fora do contexto de AutoML.

    Returns:
        (scores_df, threshold, train_errors)
        - scores_df: DataFrame com colunas [error, is_anomaly]
        - threshold: Valor do threshold calculado
        - train_errors: Array de erros no conjunto de treino
    """
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

    if threshold <= 0 or np.isnan(threshold):
        raise ValueError(f"Threshold inválido: {threshold}")

    scores_df = score_test_set(
        model, full_df, threshold=threshold, batch_size=batch_size, device=device
    )
    return scores_df, threshold, train_errors


# ── Grid builder ───────────────────────────────────────────────────────────────

def build_trials(
    equipment_id: str,
    *,
    models: list[ModelType] | None = None,
    presets: list[PresetName] | None = None,
    thresholds: list[float] | None = None,
    val_start_dates: list[datetime | None] | None = None,
    dense_layers: list[tuple[int, ...] | None] | None = None,
    dense_lrs: list[float] | None = None,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
    lstm_hidden_dims: list[int] | None = None,
    lstm_num_layers: list[int] | None = None,
    ocsvm_nus: list[float] | None = None,
    ocsvm_gammas: list[str] | None = None,
    epochs: int = 100,
    patience: int = 10,
    quick: bool = False,
) -> list[TrialConfig]:
    """
    Gera a grade de TrialConfig para um equipamento.

    Args:
        equipment_id: ID do equipamento em EQUIPMENT_CONFIGS
        models: Lista de modelos a incluir (default: todos)
        presets: Lista de presets de preprocessing (default: todos disponíveis)
        thresholds: Percentis de threshold (default: depende do modo quick)
        val_start_dates: Datas de início da validação (default: valor da config ou None)
        dense_layers: Arquiteturas de camadas densas
        dense_lrs: Learning rates para modelos densos
        batch_sizes: Tamanhos de batch
        seq_lens: Comprimentos de sequência para LSTM
        lstm_hidden_dims: Dimensões hidden do LSTM
        lstm_num_layers: Número de camadas LSTM
        ocsvm_nus: Parâmetro nu do OCSVM
        ocsvm_gammas: Parâmetro gamma do OCSVM
        epochs: Número de epochs de treino
        patience: Early stopping patience
        quick: Se True, usa grade reduzida para validação rápida

    Returns:
        Lista de TrialConfig prontos para execução
    """
    config = EQUIPMENT_CONFIGS[equipment_id]
    available_presets = (
        list(config.preprocess_presets.keys())
        if getattr(config, "preprocess_presets", None)
        else ["baseline"]
    )
    default_val_starts: list[datetime | None] = (
        [config.val_start_date] if getattr(config, "val_start_date", None) else [None]
    )

    if quick:
        _models = models or ["dense", "ocsvm"]
        _presets = presets or available_presets[:1]
        _thresholds = thresholds or [95.0]
        _val_starts = val_start_dates or default_val_starts
        _layers = dense_layers or [None]
        _lrs = dense_lrs or [1e-3]
        _batches = batch_sizes or [256]
        _seq_lens = seq_lens or [24]
        _hidden = lstm_hidden_dims or [64]
        _nlayers = lstm_num_layers or [2]
        _nus = ocsvm_nus or [0.05]
        _gammas = ocsvm_gammas or ["scale"]
        _epochs, _patience = 20, 5
    else:
        _models = models or ["dense", "lstm", "ocsvm"]
        _presets = presets or available_presets
        _thresholds = thresholds or [90.0, 95.0, 97.5, 99.0]
        _val_starts = val_start_dates or default_val_starts
        _layers = dense_layers or [None, (64, 32, 16), (128, 64, 32)]
        _lrs = dense_lrs or [1e-3, 5e-4]
        _batches = batch_sizes or [256, 512]
        _seq_lens = seq_lens or [12, 24]
        _hidden = lstm_hidden_dims or [64]
        _nlayers = lstm_num_layers or [2]
        _nus = ocsvm_nus or [0.01, 0.05, 0.1]
        _gammas = ocsvm_gammas or ["scale", "auto"]
        _epochs, _patience = epochs, patience

    trials: list[TrialConfig] = []

    for val_start, preset, model, threshold in product(_val_starts, _presets, _models, _thresholds):
        if model == "dense":
            for lr, bs, layers in product(_lrs, _batches, _layers):
                trials.append(TrialConfig(
                    preset=preset,
                    model=model,
                    threshold_percentile=threshold,
                    val_start=val_start,
                    learning_rate=lr,
                    batch_size=bs,
                    dense_layers=layers,
                    epochs=_epochs,
                    patience=_patience,
                ))

        elif model == "lstm":
            for sl, hd, nl in product(_seq_lens, _hidden, _nlayers):
                trials.append(TrialConfig(
                    preset=preset,
                    model=model,
                    threshold_percentile=threshold,
                    val_start=val_start,
                    seq_len=sl,
                    lstm_hidden_dim=hd,
                    lstm_num_layers=nl,
                    epochs=_epochs,
                    patience=_patience,
                ))

        elif model == "ocsvm":
            for nu, gamma in product(_nus, _gammas):
                trials.append(TrialConfig(
                    preset=preset,
                    model=model,
                    threshold_percentile=threshold,
                    val_start=val_start,
                    ocsvm_nu=nu,
                    ocsvm_gamma=gamma,
                ))

        else:
            raise ValueError(f"Modelo desconhecido: {model}")

    return trials


# ── Trial runner ───────────────────────────────────────────────────────────────

def run_trial(
    trial: TrialConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    full_df: pd.DataFrame,
    device: str,
    failure_date: datetime | None = None,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
    logger=None,
    label_prefix: str = "",
    trial_idx: int = 0,
) -> dict[str, Any] | None:
    """
    Executa um trial completo: treino → scoring → métricas.

    Args:
        trial: Configuração do trial
        train_df: Dados de treino pré-processados
        val_df: Dados de validação pré-processados
        full_df: Dados completos para scoring
        device: "cuda" ou "cpu"
        failure_date: Data de falha (se disponível) para métricas contextuais
        prefailure_days: Janela pré-falha em dias
        normal_end_days: Fim do período normal (dias antes da falha)
        logger: Logger do ClearML
        label_prefix: Prefixo para métricas no logger
        trial_idx: Índice do trial (para logging)

    Returns:
        Dict com métricas e objetos (_model, _scores_df) ou None se dados insuficientes
    """
    min_rows = trial.seq_len + 1 if trial.model == "lstm" else 50

    if len(train_df) < min_rows or len(val_df) < min_rows:
        return None

    model, val_loss = train_model(
        trial.model,
        train_df,
        val_df,
        device,
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
        logger=logger,
    )

    scores_df, threshold, train_errors = score_full(
        model,
        trial.model,
        train_df,
        full_df,
        trial.threshold_percentile,
        device,
        seq_len=trial.seq_len,
        batch_size=trial.batch_size,
    )

    n_anomalies = int(scores_df["is_anomaly"].sum())
    anomaly_rate = float(scores_df["is_anomaly"].mean())

    if failure_date is not None:
        metrics = failure_detection_metrics(
            scores_df,
            failure_date=failure_date,
            prefailure_days=prefailure_days,
            normal_end_days=normal_end_days,
        )
        composite_score = float(metrics["composite_score"])
    else:
        metrics = {
            "composite_score": anomaly_rate,
            "discrimination_ratio": 0.0,
            "prefailure_alert_rate": 0.0,
            "normal_alert_rate": anomaly_rate,
            "n_prefailure_alerts": 0,
            "n_normal_alerts": n_anomalies,
            "n_prefailure_samples": 0,
            "n_normal_samples": len(scores_df),
        }
        composite_score = anomaly_rate

    if logger:
        prefix = f"{label_prefix}/" if label_prefix else ""
        series = f"{prefix}automl/{trial.model}"
        logger.report_scalar(series, "val_loss", val_loss, trial_idx)
        logger.report_scalar(series, "composite_score", composite_score, trial_idx)
        logger.report_scalar(series, "n_anomalies", n_anomalies, trial_idx)
        logger.report_scalar(series, "discrimination_ratio", float(metrics["discrimination_ratio"]), trial_idx)
        logger.report_scalar(series, "prefailure_alert_rate", float(metrics["prefailure_alert_rate"]), trial_idx)
        logger.report_scalar(series, "normal_alert_rate", float(metrics["normal_alert_rate"]), trial_idx)

    row = trial.to_dict()
    row.update({
        "trial_label": trial.label(),
        "val_loss": val_loss,
        "threshold": threshold,
        "train_score_mean": float(train_errors.mean()),
        "train_score_std": float(train_errors.std()),
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "scored_samples": len(scores_df),
        "composite_score": composite_score,
        "pct_anomalies": anomaly_rate,
        "_model": model,
        "_scores_df": scores_df,
    })
    row.update(metrics)

    return row


# ── Ranking ────────────────────────────────────────────────────────────────────

def rank_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Ordena trials por composite_score (primário), prefailure_alert_rate, normal_alert_rate."""
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["composite_score", "prefailure_alert_rate", "normal_alert_rate"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


# ── Preprocessing cache ────────────────────────────────────────────────────────

def build_preprocessing_cache(
    equipment_id: str,
    splits: dict[str, pd.DataFrame],
    presets: list[str],
) -> dict[str, PreprocessedData]:
    """
    Pré-processa todos os presets de uma vez para evitar reprocessamento.

    Args:
        equipment_id: ID do equipamento
        splits: Dict com keys 'train', 'val', 'test'
        presets: Lista de nomes de presets

    Returns:
        Dict mapeando preset -> PreprocessedData
    """
    cache: dict[str, PreprocessedData] = {}

    full_raw = pd.concat([
        splits["train"],
        splits["val"],
        splits["test"],
    ]).sort_index()

    for preset in presets:
        print(f"  Preprocessing preset='{preset}'...")

        steps = get_preprocessing_steps(equipment_id, preset=preset)

        train_df, artifacts, _ = run_preprocessing(
            splits["train"], steps, return_artifacts=True, return_report=True,
        )
        val_df, _, _ = run_preprocessing(
            splits["val"], steps, fitted_artifacts=artifacts,
            return_artifacts=True, return_report=True,
        )
        full_df, _, _ = run_preprocessing(
            full_raw, steps, fitted_artifacts=artifacts,
            return_artifacts=True, return_report=True,
        )

        cache[preset] = PreprocessedData(
            train_df=train_df,
            val_df=val_df,
            full_df=full_df,
            artifacts=artifacts,
        )

        print(f"  Train: {train_df.shape} | Val: {val_df.shape} | Full: {full_df.shape}")

    return cache


# ── Relatórios formatados ──────────────────────────────────────────────────────

def print_trial_header(trial_num: int, total_trials: int, trial: TrialConfig) -> None:
    """Imprime cabeçalho de um trial."""
    print(f"\n[{trial_num:03d}/{total_trials:03d}] {trial.label()}")
    print("─" * 70)


def print_trial_result(result: dict[str, Any], indent: str = "  ") -> None:
    """Imprime resultado formatado de um trial."""
    # — Treino
    if result.get("val_loss", 0.0) > 0.0:
        print(f"{indent}Val Loss   : {result['val_loss']:.5f}")
    print(f"{indent}Threshold  : {result['threshold']:.5f}  "
          f"(mu={result['train_score_mean']:.5f} +-{result['train_score_std']:.5f})")

    # — Anomalias globais
    print(f"{indent}Anomalias  : {result['n_anomalies']}/{result['scored_samples']} "
          f"({result['anomaly_rate']:.2%})")

    # — Metricas de deteccao de falha
    pre_rate = result.get("prefailure_alert_rate", 0.0)
    norm_rate = result.get("normal_alert_rate", 0.0)
    pre_n = result.get("n_prefailure_alerts", 0)
    pre_total = result.get("n_prefailure_samples", 0)
    norm_n = result.get("n_normal_alerts", 0)
    norm_total = result.get("n_normal_samples", 0)

    if pre_total > 0 or norm_total > 0:
        print(f"{indent}Pre-falha  : {pre_n}/{pre_total} ({pre_rate:.2%})"
              f"  |  Normal: {norm_n}/{norm_total} ({norm_rate:.2%})")

    ratio = result.get("discrimination_ratio", 0.0)
    if ratio > 0:
        print(f"{indent}Ratio Disc.: {ratio:.3f}")

    # — Score composto (sempre por ultimo, e o primario)
    print(f"{indent}Score      : {result['composite_score']:.5f}")


def print_final_summary(
    equipment_id: str,
    n_total_trials: int,
    n_successful: int,
    n_failed: int,
    n_skipped: int,
    best_result: dict[str, Any],
    duration_seconds: float,
) -> None:
    """Imprime resumo final da execução do AutoML."""
    print("\n" + "=" * 70)
    print(f"{'RESUMO FINAL':^70}")
    print("=" * 70)
    print(f"Equipamento: {equipment_id}")
    print(f"Trials executados: {n_successful}/{n_total_trials} bem-sucedidos")
    print(f"Falhas: {n_failed}")
    print(f"Pulados: {n_skipped}")
    print(f"Duração: {duration_seconds / 60:.1f} min")
    print()
    print("MELHOR CONFIGURAÇÃO:")
    print(f"  Modelo    : {best_result['model']}")
    print(f"  Preset    : {best_result['preset']}")
    print(f"  Val Start : {best_result.get('val_start') or 'auto'}")
    print(f"  Score     : {best_result['composite_score']:.5f}")
    print(f"  Val Loss  : {best_result['val_loss']:.5f}")
    print(f"  Threshold : {best_result['threshold']:.5f}")
    print(f"  Anomalias : {best_result['n_anomalies']}/{best_result['scored_samples']}")
    print("=" * 70 + "\n")


def print_ranking_table(ranking: pd.DataFrame, top_n: int = 10) -> None:
    """
    Imprime tabela final de resultados com duas seções:
    - Top-N geral (todos os modelos, ordenado por composite_score)
    - Top-3 por modelo (melhor de cada tipo)
    """
    W = 100

    # Colunas base sempre presentes
    base_cols = [
        "model", "preset", "val_start",
        "composite_score", "prefailure_alert_rate", "normal_alert_rate",
        "discrimination_ratio", "val_loss", "threshold",
        "n_anomalies", "scored_samples",
    ]
    cols = [c for c in base_cols if c in ranking.columns]

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df[cols].copy()
        for c in ("composite_score", "prefailure_alert_rate", "normal_alert_rate", "val_loss", "threshold"):
            if c in out:
                out[c] = out[c].apply(lambda v: f"{v:.4f}")
        if "discrimination_ratio" in out:
            out["discrimination_ratio"] = out["discrimination_ratio"].apply(lambda v: f"{v:.2f}")
        if "val_start" in out:
            out["val_start"] = out["val_start"].fillna("auto")
        return out

    # ── Top-N geral ───────────────────────────────────────────────
    print("\n" + "=" * W)
    print(f"{'RANKING FINAL — TOP ' + str(top_n) + ' GERAL':^{W}}")
    print("=" * W)
    print(_fmt(ranking.head(top_n)).to_string(index=True))

    # ── Top-3 por modelo ──────────────────────────────────────────
    if "model" in ranking.columns:
        print("\n" + "=" * W)
        print(f"{'RANKING POR MODELO — TOP 3 DE CADA':^{W}}")
        print("=" * W)

        for model_type in sorted(ranking["model"].unique()):
            subset = ranking[ranking["model"] == model_type].head(3)
            print(f"\n  [{model_type.upper()}]")
            print("  " + "-" * (W - 2))
            print(_fmt(subset).to_string(index=True).replace("\n", "\n  "))

    print("\n" + "=" * W + "\n")


# ── Salvamento de artifacts ────────────────────────────────────────────────────

def save_artifacts(
    best_row: dict[str, Any],
    ranking: pd.DataFrame,
    equipment_id: str,
    output_dir: Path,
    task: Task,
    upload_to_clearml: bool = True,
) -> None:
    """
    Salva todos os artifacts localmente e opcionalmente faz upload ao ClearML.

    Args:
        best_row: Dict do melhor trial (deve conter _model, _scores_df, _artifacts)
        ranking: DataFrame com ranking de todos os trials
        equipment_id: ID do equipamento
        output_dir: Diretório de saída local
        task: Task do ClearML
        upload_to_clearml: Se True, faz upload ao ClearML
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model = best_row["_model"]
    model_type = best_row["model"]

    if model_type == "ocsvm":
        model_path = output_dir / f"best_model_{equipment_id}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
    else:
        model_path = output_dir / f"best_model_{equipment_id}.pt"
        torch.save(model.state_dict(), model_path)

    print(f"  Modelo salvo: {model_path}")

    scores_path = output_dir / "best_full_scores.parquet"
    best_row["_scores_df"].to_parquet(scores_path)
    print(f"  Scores salvos: {scores_path}")

    ranking_path = output_dir / "automl_ranking.parquet"
    ranking.to_parquet(ranking_path)
    print(f"  Ranking salvo: {ranking_path}")

    if "_artifacts" in best_row:
        artifacts_path = output_dir / "preprocessing_artifacts.pkl"
        with artifacts_path.open("wb") as f:
            pickle.dump(best_row["_artifacts"], f)
        print(f"  Artifacts salvos: {artifacts_path}")

    summary = {k: v for k, v in best_row.items() if not k.startswith("_")}
    summary_path = output_dir / "best_trial_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary salvo: {summary_path}")

    if upload_to_clearml:
        print("  Fazendo upload ao ClearML...")
        task.upload_artifact("best_model", artifact_object=model_path)
        task.upload_artifact("best_full_scores", artifact_object=best_row["_scores_df"])
        task.upload_artifact("automl_results", artifact_object=ranking)
        task.upload_artifact("best_trial", artifact_object={"results": summary})

        if "_artifacts" in best_row:
            task.upload_artifact("preprocessing_artifacts", artifact_object=best_row["_artifacts"])

        print("  ✓ Upload completo")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    equipment_id: str,
    remote: bool = False,
    local_data: bool = False,
    queue: str = "default",
    upload_to_clearml: bool = True,
    local_artifacts_dir: str = "artifacts_local",
    models: list[str] | None = None,
    quick: bool = False,
    epochs: int = 100,
    patience: int = 10,
    prefailure_days: int = 30,
    normal_end_days: int = 60,
) -> None:
    """
    Pipeline principal do AutoML.

    Etapas:
    1. Setup: ClearML, device, configuração
    2. Data Loading: carrega e pré-processa dados
    3. Splitting: divisão temporal train/val/test
    4. Trial Generation: monta grade de busca
    5. Preprocessing Cache: pré-processa todos os presets
    6. Execution: executa trials
    7. Reporting: salva resultados e artifacts
    """
    start_time = time.time()

    # ══════════════════════════════════════════════════════════════
    # ETAPA 1: SETUP
    # ══════════════════════════════════════════════════════════════

    config = EQUIPMENT_CONFIGS[equipment_id]

    print("=" * 70)
    print(f"{'AUTOML - DETECÇÃO DE ANOMALIAS':^70}")
    print("=" * 70)
    print(f"Equipamento: {equipment_id}")
    print(f"Quick      : {quick}")
    print(f"Remote     : {remote}")
    print("=" * 70 + "\n")

    Task.add_requirements("pyarrow")
    Task.add_requirements("torch", package_version="")

    speed_suffix = "quick" if quick else "full"
    task_name = f"automl_{equipment_id}_multivariate_{speed_suffix}"

    task = Task.init(
        project_name="Transpetro",
        task_name=task_name,
        output_uri=True,
    )
    task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")

    hparams = {
        "equipment_id": equipment_id,
        "models": models or ["dense", "lstm", "ocsvm"],
        "quick": quick,
        "epochs": epochs,
        "patience": patience,
        "prefailure_days": prefailure_days,
        "normal_end_days": normal_end_days,
        "upload_to_clearml": upload_to_clearml,
        "failure_date": str(getattr(config, "failure_date", None)),
    }
    task.connect(hparams)

    if remote:
        print(f"Executando remotamente na fila: {queue}")
        task.execute_remotely(queue_name=queue)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = task.get_logger()

    print(f"Device: {device}")
    print(f"ClearML Task ID: {task.id}\n")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 2: DATA LOADING
    # ══════════════════════════════════════════════════════════════

    print("ETAPA 2: Carregando dados...")
    df = load_equipment_data(equipment_id, from_clearml=not local_data)
    print(f"  Shape inicial: {df.shape}")
    print(f"  Período: {df.index.min()} até {df.index.max()}")

    if getattr(config, "pre_split_steps", None):
        print("  Aplicando pré-split preprocessing...")
        df, _, _, _ = run_preprocessing(df, config.pre_split_steps, return_report=True)
        print(f"  Shape pós pré-split: {df.shape}")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 3: TEMPORAL SPLITTING
    # ══════════════════════════════════════════════════════════════

    print("\nETAPA 3: Split temporal...")

    split_kwargs = {}
    if hasattr(config, "val_start_date") and config.val_start_date:
        split_kwargs["val_start_date"] = config.val_start_date
    if hasattr(config, "val_end_date") and config.val_end_date:
        split_kwargs["val_end_date"] = config.val_end_date

    splits = temporal_split(df, **split_kwargs)

    print(f"  Train: {splits['train'].shape} [{splits['train'].index.min()} → {splits['train'].index.max()}]")
    print(f"  Val  : {splits['val'].shape} [{splits['val'].index.min()} → {splits['val'].index.max()}]")
    print(f"  Test : {splits['test'].shape} [{splits['test'].index.min()} → {splits['test'].index.max()}]")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 4: TRIAL GENERATION
    # ══════════════════════════════════════════════════════════════

    print("\nETAPA 4: Gerando grid de trials...")
    trials = build_trials(
        equipment_id=equipment_id,
        models=hparams["models"],
        epochs=epochs,
        patience=patience,
        quick=quick,
    )
    print(f"  Total de trials: {len(trials)}")

    model_counts: dict[str, int] = {}
    for trial in trials:
        model_counts[trial.model] = model_counts.get(trial.model, 0) + 1

    for model_type, count in sorted(model_counts.items()):
        print(f"    {model_type:8s}: {count:4d} trials")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 5: PREPROCESSING CACHE
    # ══════════════════════════════════════════════════════════════

    print("\nETAPA 5: Preparando cache de preprocessing...")
    presets = sorted(set(t.preset for t in trials))
    preprocessing_cache = build_preprocessing_cache(equipment_id, splits, presets)
    print(f"  ✓ {len(preprocessing_cache)} presets em cache\n")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 6: EXECUTION
    # ══════════════════════════════════════════════════════════════

    print("=" * 70)
    print(f"{'EXECUÇÃO DE TRIALS':^70}")
    print("=" * 70)

    rows: list[dict] = []
    best_row: dict | None = None
    n_skipped = 0
    n_failed = 0

    failure_date = getattr(config, "failure_date", None)

    for i, trial in enumerate(trials, 1):
        print_trial_header(i, len(trials), trial)

        pdata = preprocessing_cache[trial.preset]

        try:
            row = run_trial(
                trial=trial,
                train_df=pdata.train_df,
                val_df=pdata.val_df,
                full_df=pdata.full_df,
                device=device,
                failure_date=failure_date,
                prefailure_days=prefailure_days,
                normal_end_days=normal_end_days,
                logger=logger,
                trial_idx=i,
            )

        except Exception as e:
            print(f"  [ERRO] {type(e).__name__}: {e}")
            n_failed += 1
            continue

        if row is None:
            print("  [SKIP] Dados insuficientes")
            n_skipped += 1
            continue

        row["_artifacts"] = pdata.artifacts
        print_trial_result(row)

        if best_row is None or row["composite_score"] > best_row["composite_score"]:
            if best_row is not None:
                best_row.pop("_model", None)
                best_row.pop("_scores_df", None)
            best_row = row
        else:
            row.pop("_model", None)
            row.pop("_scores_df", None)

        rows.append({k: v for k, v in row.items() if not k.startswith("_")})

    if best_row is None:
        raise RuntimeError("Todos os trials falharam.")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 7: REPORTING
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print(f"{'SALVANDO RESULTADOS':^70}")
    print("=" * 70)

    ranking = rank_results(rows)

    print_ranking_table(ranking)

    output_dir = Path(local_artifacts_dir) / f"{task.id}_{equipment_id}"
    save_artifacts(
        best_row=best_row,
        ranking=ranking,
        equipment_id=equipment_id,
        output_dir=output_dir,
        task=task,
        upload_to_clearml=upload_to_clearml,
    )

    logger.report_scalar("best", "composite_score", best_row["composite_score"], 0)
    logger.report_scalar("best", "val_loss", best_row["val_loss"], 0)
    logger.report_scalar("best", "threshold", best_row["threshold"], 0)
    logger.report_scalar("best", "n_anomalies", best_row["n_anomalies"], 0)

    duration = time.time() - start_time
    n_successful = len(rows)

    print_final_summary(
        equipment_id=equipment_id,
        n_total_trials=len(trials),
        n_successful=n_successful,
        n_failed=n_failed,
        n_skipped=n_skipped,
        best_result=best_row,
        duration_seconds=duration,
    )

    print(f"Artifacts salvos em: {output_dir}")
    if upload_to_clearml:
        print(f"Artifacts também enviados ao ClearML (Task ID: {task.id})")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoML para detecção de anomalias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Teste rápido local
  python scripts/automl_anomaly_v3.py --equipment MEQ-01 --local-data --quick

  # Execução completa local
  python scripts/automl_anomaly_v3.py --equipment MEQ-01 --local-data

  # Execução remota no ClearML
  python scripts/automl_anomaly_v3.py --equipment MEQ-01 --remote --queue gpu

  # Custom: apenas Dense e OCSVM
  python scripts/automl_anomaly_v3.py --equipment MEQ-01 --models dense ocsvm
        """,
    )

    parser.add_argument(
        "--equipment",
        required=True,
        choices=list(EQUIPMENT_CONFIGS.keys()),
        help="ID do equipamento a treinar",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Executa remotamente no ClearML",
    )
    parser.add_argument(
        "--queue",
        default="default",
        help="Fila ClearML para execução remota (default: default)",
    )
    parser.add_argument(
        "--local-data",
        action="store_true",
        help="Carrega dados locais em vez do ClearML Dataset",
    )
    parser.add_argument(
        "--no-clearml-upload",
        action="store_true",
        help="Desabilita upload de artifacts ao ClearML",
    )
    parser.add_argument(
        "--local-artifacts-dir",
        default="artifacts_local",
        help="Diretório para artifacts locais (default: artifacts_local)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["dense", "lstm", "ocsvm"],
        help="Modelos a incluir (padrão: todos)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grade reduzida: 20 epochs, 1 preset, 1 threshold — para validar rápido",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de epochs de treino (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--prefailure-days",
        type=int,
        default=30,
        help="Dias antes da falha que compõem a janela pré-falha (default: 30)",
    )
    parser.add_argument(
        "--normal-end-days",
        type=int,
        default=60,
        help="Dias antes da falha onde termina o período normal (default: 60)",
    )

    args = parser.parse_args()

    main(
        equipment_id=args.equipment,
        remote=args.remote,
        local_data=args.local_data,
        queue=args.queue,
        upload_to_clearml=not args.no_clearml_upload,
        local_artifacts_dir=args.local_artifacts_dir,
        models=args.models,
        quick=args.quick,
        epochs=args.epochs,
        patience=args.patience,
        prefailure_days=args.prefailure_days,
        normal_end_days=args.normal_end_days,
    )