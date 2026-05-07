"""
AutoML para Detecção de Anomalias
==================================
Modelos : Dense Autoencoder | LSTM Autoencoder | One-Class SVM
Seleção : Score composto (val_loss + anomaly_rate + threshold_ratio)
Busca   : Grid search declarativo via build_trials()
Modos   : Multivariado e Per-sensor (--per-sensor)

Uso rápido:
    python scripts/automl_anomaly.py --equipment MEQ-01 --local-data --quick
Uso completo:
    python scripts/automl_anomaly.py --equipment MEQ-01 --remote --queue gpu
"""

import argparse
import pickle
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

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


# ── TrialConfig ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrialConfig:
    preset: str
    model: str
    threshold_percentile: float
    # Dense
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10
    dense_layers: tuple[int, ...] | None = None
    # LSTM
    seq_len: int = 24
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    # OCSVM
    ocsvm_nu: float = 0.05
    ocsvm_gamma: str = "scale"

    def label(self) -> str:
        parts = [self.preset, self.model, f"p{self.threshold_percentile:g}"]
        if self.model == "dense":
            parts += [f"lr{self.learning_rate:g}", f"b{self.batch_size}"]
            if self.dense_layers:
                parts.append("layers" + "-".join(str(v) for v in self.dense_layers))
        elif self.model == "lstm":
            parts += [f"seq{self.seq_len}", f"h{self.lstm_hidden_dim}", f"l{self.lstm_num_layers}"]
        else:
            parts += [f"nu{self.ocsvm_nu:g}", f"g{self.ocsvm_gamma}"]
        return "__".join(parts)


# ── Grid builder ───────────────────────────────────────────────────────────────

def build_trials(
    equipment_id: str,
    *,
    models: list[str] | None = None,
    presets: list[str] | None = None,
    thresholds: list[float] | None = None,
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
    Com quick=True usa grade reduzida para validação rápida.
    """
    config = EQUIPMENT_CONFIGS[equipment_id]
    available_presets = (
        list(config.preprocess_presets.keys())
        if getattr(config, "preprocess_presets", None)
        else ["baseline"]
    )

    if quick:
        _models     = models     or ["dense", "ocsvm"]
        _presets    = presets    or available_presets[:1]
        _thresholds = thresholds or [95.0]
        _layers     = dense_layers or [None]
        _lrs        = dense_lrs  or [1e-3]
        _batches    = batch_sizes or [256]
        _seq_lens   = seq_lens   or [24]
        _hidden     = lstm_hidden_dims or [64]
        _nlayers    = lstm_num_layers  or [2]
        _nus        = ocsvm_nus  or [0.05]
        _gammas     = ocsvm_gammas or ["scale"]
        _epochs, _patience = 20, 5
    else:
        _models     = models     or ["dense", "lstm", "ocsvm"]
        _presets    = presets    or available_presets
        _thresholds = thresholds or [90.0, 95.0, 97.5, 99.0]
        _layers     = dense_layers or [None, (64, 32, 16), (128, 64, 32)]
        _lrs        = dense_lrs  or [1e-3, 5e-4]
        _batches    = batch_sizes or [256, 512]
        _seq_lens   = seq_lens   or [12, 24]
        _hidden     = lstm_hidden_dims or [64]
        _nlayers    = lstm_num_layers  or [2]
        _nus        = ocsvm_nus  or [0.01, 0.05, 0.1]
        _gammas     = ocsvm_gammas or ["scale", "auto"]
        _epochs, _patience = epochs, patience

    trials: list[TrialConfig] = []
    for preset, model, threshold in product(_presets, _models, _thresholds):
        if model == "dense":
            for lr, bs, layers in product(_lrs, _batches, _layers):
                trials.append(TrialConfig(
                    preset=preset, model=model, threshold_percentile=threshold,
                    learning_rate=lr, batch_size=bs,
                    dense_layers=layers, epochs=_epochs, patience=_patience,
                ))
        elif model == "lstm":
            for sl, hd, nl in product(_seq_lens, _hidden, _nlayers):
                trials.append(TrialConfig(
                    preset=preset, model=model, threshold_percentile=threshold,
                    seq_len=sl, lstm_hidden_dim=hd, lstm_num_layers=nl,
                    epochs=_epochs, patience=_patience,
                ))
        elif model == "ocsvm":
            for nu, gamma in product(_nus, _gammas):
                trials.append(TrialConfig(
                    preset=preset, model=model, threshold_percentile=threshold,
                    ocsvm_nu=nu, ocsvm_gamma=gamma,
                ))
        else:
            raise ValueError(f"Modelo desconhecido: {model}")

    return trials


# ── Treino genérico ────────────────────────────────────────────────────────────

def train_model(
    trial: TrialConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
    logger=None,
) -> tuple[Any, float]:
    """Treina e retorna (model, best_val_loss). OCSVM retorna (clf, 0.0)."""
    if trial.model == "ocsvm":
        return fit_ocsvm(train_df, nu=trial.ocsvm_nu, gamma=trial.ocsvm_gamma), 0.0

    n_features = train_df.shape[1]

    if trial.model == "lstm":
        model = LSTMAutoencoder(
            input_dim=n_features,
            hidden_dim=trial.lstm_hidden_dim,
            num_layers=trial.lstm_num_layers,
            seq_len=trial.seq_len,
        ).to(device)
        train_loader = make_sequence_dataloader(
            train_df, seq_len=trial.seq_len, batch_size=trial.batch_size,
            shuffle=True, device=device,
        )
        val_loader = make_sequence_dataloader(
            val_df, seq_len=trial.seq_len, batch_size=trial.batch_size,
            shuffle=False, device=device,
        )
    else:
        model = DenseAutoencoder(
            input_dim=n_features,
            encoding_layers=list(trial.dense_layers) if trial.dense_layers else None,
        ).to(device)
        train_loader = make_dataloader(
            train_df, batch_size=trial.batch_size, shuffle=True, device=device
        )
        val_loader = make_dataloader(
            val_df, batch_size=trial.batch_size, shuffle=False, device=device
        )

    model, best_val_loss = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=trial.epochs,
        learning_rate=trial.learning_rate,
        weight_decay=trial.weight_decay,
        patience=trial.patience,
        logger=logger,
        return_val_loss=True,
    )
    return model, float(best_val_loss)


# ── Scoring genérico ───────────────────────────────────────────────────────────

def score_full(
    model,
    trial: TrialConfig,
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    device: str,
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """
    Calcula threshold no train_df e aplica sobre full_df.
    Retorna (scores_df, threshold, train_errors).
    """
    bs = trial.batch_size

    if trial.model == "ocsvm":
        train_errors = compute_ocsvm_errors(model, train_df)
        threshold = determine_threshold(train_errors, percentile=trial.threshold_percentile)
        return score_ocsvm_set(model, full_df, threshold), threshold, train_errors

    if trial.model == "lstm":
        train_errors = compute_reconstruction_errors_sequence(
            model, train_df, seq_len=trial.seq_len, batch_size=bs, device=device,
        )
        threshold = determine_threshold(train_errors, percentile=trial.threshold_percentile)
        scores = score_test_set_sequence(
            model, full_df, seq_len=trial.seq_len,
            threshold=threshold, batch_size=bs, device=device,
        )
        return scores, threshold, train_errors

    train_errors = compute_reconstruction_errors(model, train_df, batch_size=bs, device=device)
    threshold = determine_threshold(train_errors, percentile=trial.threshold_percentile)
    scores = score_test_set(model, full_df, threshold=threshold, batch_size=bs, device=device)
    return scores, threshold, train_errors


# ── Score composto ─────────────────────────────────────────────────────────────

def composite_score(
    val_loss: float,
    train_errors: np.ndarray,
    scores_df: pd.DataFrame,
    threshold: float,
    w_val: float = 0.4,
    w_anomaly_rate: float = 0.3,
    w_threshold_ratio: float = 0.3,
) -> float:
    """
    Score composto — quanto MENOR, melhor.
      val_loss        : qualidade de reconstrução no val set
      anomaly_rate    : penaliza hipersensibilidade
      threshold_ratio : penaliza limiares instáveis (threshold / train_mean)
    """
    anomaly_rate = float(scores_df["is_anomaly"].mean())
    threshold_ratio = threshold / max(float(train_errors.mean()), 1e-9)
    return w_val * val_loss + w_anomaly_rate * anomaly_rate + w_threshold_ratio * threshold_ratio


# ── Runner de um trial ─────────────────────────────────────────────────────────

def run_trial(
    trial: TrialConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    full_df: pd.DataFrame,
    device: str,
    logger=None,
    label_prefix: str = "",
    trial_idx: int = 0,
) -> dict[str, Any] | None:
    """
    Executa um trial completo. Retorna None se dados insuficientes.
    Campos '_model' e '_scores_df' ficam no dict para uso posterior.
    """
    min_rows = trial.seq_len + 1 if trial.model == "lstm" else 50
    if len(train_df) < min_rows or len(val_df) < min_rows:
        print(f"    [SKIP] dados insuficientes (train={len(train_df)}, val={len(val_df)})")
        return None

    model, val_loss = train_model(trial, train_df, val_df, device)
    scores_df, threshold, train_errors = score_full(model, trial, train_df, full_df, device)
    score = composite_score(val_loss, train_errors, scores_df, threshold)
    n_anomalies = int(scores_df["is_anomaly"].sum())

    if logger:
        prefix = f"{label_prefix}/" if label_prefix else ""
        logger.report_scalar(f"{prefix}automl/{trial.model}", "val_loss", val_loss, trial_idx)
        logger.report_scalar(f"{prefix}automl/{trial.model}", "composite_score", score, trial_idx)
        logger.report_scalar(f"{prefix}automl/{trial.model}", "n_anomalies", n_anomalies, trial_idx)

    row = asdict(trial)
    row["dense_layers"] = (
        "auto" if trial.dense_layers is None
        else ",".join(str(v) for v in trial.dense_layers)
    )
    row.update({
        "trial_label": trial.label(),
        "val_loss": val_loss,
        "threshold": threshold,
        "train_score_mean": float(train_errors.mean()),
        "train_score_std": float(train_errors.std()),
        "n_anomalies": n_anomalies,
        "anomaly_rate": float(scores_df["is_anomaly"].mean()),
        "scored_samples": len(scores_df),
        "composite_score": score,
        "_model": model,
        "_scores_df": scores_df,
    })
    return row


# ── Ranking ────────────────────────────────────────────────────────────────────

def rank_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Ordena por composite_score (menor = melhor), retorna apenas colunas de métricas."""
    metric_cols = [
        "trial_label", "model", "preset", "composite_score",
        "val_loss", "threshold", "train_score_mean", "anomaly_rate", "n_anomalies",
        "threshold_percentile", "learning_rate", "batch_size", "dense_layers",
        "seq_len", "lstm_hidden_dim", "lstm_num_layers", "ocsvm_nu", "ocsvm_gamma",
    ]
    df = pd.DataFrame(rows)
    available = [c for c in metric_cols if c in df.columns]
    return df[available].sort_values("composite_score", ascending=True).reset_index(drop=True)


# ── AutoML principal ───────────────────────────────────────────────────────────

def run_automl(
    trials: list[TrialConfig],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    full_df: pd.DataFrame,
    device: str,
    logger=None,
    label_prefix: str = "",
) -> tuple[dict, list[dict]]:
    """
    Executa todos os trials. Retorna (melhor_row, todos_os_rows).
    '_model' e '_scores_df' ficam apenas no melhor_row para economizar memória.
    """
    rows: list[dict] = []
    best_row: dict | None = None

    print(f"\n{'='*60}")
    print(f"  AutoML: {len(trials)} trials | prefix='{label_prefix}'")
    print(f"{'='*60}")

    for i, trial in enumerate(trials, 1):
        print(f"  [{i}/{len(trials)}] {trial.label()}")
        try:
            row = run_trial(
                trial, train_df, val_df, full_df, device,
                logger=logger, label_prefix=label_prefix, trial_idx=i,
            )
        except Exception as e:
            print(f"    [ERRO] {e}")
            continue

        if row is None:
            continue

        print(
            f"    val_loss={row['val_loss']:.5f} | threshold={row['threshold']:.5f} | "
            f"anomalies={row['n_anomalies']}/{row['scored_samples']} | "
            f"score={row['composite_score']:.5f}"
        )

        # Mantém _model/_scores_df apenas no melhor — libera memória dos demais
        if best_row is None or row["composite_score"] < best_row["composite_score"]:
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

    print(f"\n  ✓ MELHOR: {best_row['trial_label']}")
    print(
        f"    score={best_row['composite_score']:.5f} | "
        f"val_loss={best_row['val_loss']:.5f} | "
        f"threshold={best_row['threshold']:.5f}"
    )
    return best_row, rows


# ── Publicação de artifacts ────────────────────────────────────────────────────

def _sensor_slug(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _save_local(local_dir: Path, name: str, obj) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9a-zA-Z_.-]+", "_", name)
    if isinstance(obj, (str, Path)):
        src = Path(obj)
        if src.exists() and src.is_file():
            tgt = local_dir / f"{safe}{src.suffix}"
            shutil.copy2(src, tgt)
            return tgt
    if isinstance(obj, pd.DataFrame):
        tgt = local_dir / f"{safe}.parquet"
        obj.to_parquet(tgt)
        return tgt
    tgt = local_dir / f"{safe}.pkl"
    with tgt.open("wb") as f:
        pickle.dump(obj, f)
    return tgt


def _publish(task: Task, name: str, obj, local_dir: Path, upload: bool) -> None:
    try:
        _save_local(local_dir, name, obj)
    except Exception as e:
        print(f"[WARN] salvar local '{name}': {e}")
    if upload:
        try:
            task.upload_artifact(name, artifact_object=obj)
        except Exception as e:
            print(f"[WARN] upload ClearML '{name}': {e}")


def _publish_best(
    task: Task,
    best_row: dict,
    artifacts: PreprocessingArtifacts,
    ranking: pd.DataFrame,
    equipment_id: str,
    local_dir: Path,
    upload: bool,
    slug: str = "",
    logger=None,
    label_prefix: str = "",
) -> None:
    sfx = f"__{slug}" if slug else ""
    prefix = f"{label_prefix}/" if label_prefix else ""
    model = best_row["_model"]
    scores_df = best_row["_scores_df"]

    if best_row["model"] == "ocsvm":
        _publish(task, f"model_file{sfx}", model, local_dir, upload)
    else:
        model_path = f"model_{equipment_id}{sfx}.pt"
        torch.save(model.state_dict(), model_path)
        _publish(task, f"model_file{sfx}", model_path, local_dir, upload)

    _publish(task, f"scaler{sfx}", artifacts.scaler, local_dir, upload)
    if artifacts.clip_bounds:
        _publish(task, f"clip_bounds{sfx}", artifacts.clip_bounds, local_dir, upload)
    if artifacts.knn_imputer is not None:
        _publish(task, f"knn_imputer{sfx}", artifacts.knn_imputer, local_dir, upload)

    _publish(task, f"full_scores{sfx}", scores_df, local_dir, upload)
    _publish(task, f"automl_ranking{sfx}", ranking, local_dir, upload)

    summary = {k: v for k, v in best_row.items() if not k.startswith("_")}
    _publish(task, f"results{sfx}", summary, local_dir, upload)

    if logger:
        logger.report_scalar(f"{prefix}best", "composite_score", best_row["composite_score"], 0)
        logger.report_scalar(f"{prefix}best", "val_loss", best_row["val_loss"], 0)
        logger.report_scalar(f"{prefix}best", "threshold", best_row["threshold"], 0)
        logger.report_scalar(f"{prefix}best", "n_anomalies", best_row["n_anomalies"], 0)


# ── Per-sensor ─────────────────────────────────────────────────────────────────

def run_per_sensor(
    task: Task,
    splits: dict,
    equipment_id: str,
    trials: list[TrialConfig],
    local_task_dir: Path,
    upload: bool,
    device: str,
    logger,
) -> None:
    sensors = list(splits["train"].columns)
    print(f"\nPer-sensor: {len(sensors)} sensores")
    summary_rows = []

    for sensor in sensors:
        slug = _sensor_slug(sensor)
        steps = [{"step": "select_features", "features": [sensor]}]
        print(f"\n{'─'*50}\nSensor: {sensor}")

        train_df, artifacts, _ = run_preprocessing(
            splits["train"], steps, return_artifacts=True, return_report=True
        )
        val_df, _, _ = run_preprocessing(
            splits["val"], steps, fitted_artifacts=artifacts,
            return_artifacts=True, return_report=True,
        )
        full_raw = pd.concat([splits["train"], splits["val"], splits["test"]]).sort_index()
        full_df, _, _ = run_preprocessing(
            full_raw, steps, fitted_artifacts=artifacts,
            return_artifacts=True, return_report=True,
        )

        best_row, rows = run_automl(
            trials, train_df, val_df, full_df, device,
            logger=logger, label_prefix=slug,
        )
        ranking = rank_results(rows)

        _publish_best(
            task, best_row, artifacts, ranking, equipment_id,
            local_task_dir / "per_sensor" / slug,
            upload, slug=slug, logger=logger, label_prefix=slug,
        )

        summary_rows.append({
            "sensor": sensor,
            "slug": slug,
            "best_model": best_row["model"],
            "best_preset": best_row["preset"],
            "composite_score": best_row["composite_score"],
            "val_loss": best_row["val_loss"],
            "threshold": best_row["threshold"],
            "n_anomalies": best_row["n_anomalies"],
            "n_trials": len(rows),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("sensor").reset_index(drop=True)
    _publish(task, "per_sensor_summary", summary_df, local_task_dir, upload)
    print("\n=== Per-Sensor Summary ===")
    print(summary_df.to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────────

def main(
    equipment_id: str,
    remote: bool = False,
    local_data: bool = False,
    per_sensor: bool = False,
    queue: str = "default",
    upload_to_clearml: bool = True,
    local_artifacts_dir: str = "artifacts_local",
    models: list[str] | None = None,
    quick: bool = False,
    epochs: int = 100,
    patience: int = 10,
) -> None:
    config = EQUIPMENT_CONFIGS[equipment_id]

    Task.add_requirements("pyarrow")
    Task.add_requirements("torch", package_version="")

    mode = "per-sensor" if per_sensor else "multivariate"
    task_name = f"automl-{equipment_id}-{mode}-{'quick' if quick else 'full'}"
    task = Task.init(project_name="Transpetro", task_name=task_name, output_uri=True)
    task.set_base_docker(docker_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")

    hparams: dict[str, Any] = {
        "equipment_id": equipment_id,
        "models": models or ["dense", "lstm", "ocsvm"],
        "quick": quick,
        "per_sensor": per_sensor,
        "epochs": epochs,
        "patience": patience,
        "upload_to_clearml": upload_to_clearml,
    }
    task.connect(hparams)

    if remote:
        task.execute_remotely(queue_name=queue)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Modo: {mode} | Quick: {quick}")

    # 1. Carregar dados
    df = load_equipment_data(equipment_id, from_clearml=not local_data)
    print(f"Dados: {df.shape}")

    # 2. Pré-processamento pré-split
    if getattr(config, "pre_split_steps", None):
        df, _, _ = run_preprocessing(df, config.pre_split_steps)
        print(f"Pós pré-split: {df.shape}")

    # 3. Split temporal
    val_start = getattr(config, "val_start_date", None)
    val_end = getattr(config, "val_end_date", None)
    splits = temporal_split(df, val_start_date=val_start, val_end_date=val_end)
    print(
        f"Train: {splits['train'].shape} | "
        f"Val: {splits['val'].shape} | "
        f"Test: {splits['test'].shape}"
    )

    # 4. Gerar grade de trials
    trials = build_trials(
        equipment_id,
        models=hparams["models"],
        epochs=epochs,
        patience=patience,
        quick=quick,
    )
    print(f"Trials gerados: {len(trials)}")

    logger = task.get_logger()
    local_task_dir = Path(local_artifacts_dir) / f"{task.id}_{equipment_id}"

    # 5. Rodar AutoML
    if per_sensor:
        run_per_sensor(
            task=task,
            splits=splits,
            equipment_id=equipment_id,
            trials=trials,
            local_task_dir=local_task_dir,
            upload=upload_to_clearml,
            device=device,
            logger=logger,
        )
        return

    # Modo multivariado — pré-processamento pós-split
    steps = get_preprocessing_steps(equipment_id, preset="baseline")
    train_df, artifacts, _ = run_preprocessing(
        splits["train"], steps, return_artifacts=True, return_report=True
    )
    val_df, _, _ = run_preprocessing(
        splits["val"], steps, fitted_artifacts=artifacts,
        return_artifacts=True, return_report=True,
    )
    full_raw = pd.concat([splits["train"], splits["val"], splits["test"]]).sort_index()
    full_df, _, _ = run_preprocessing(
        full_raw, steps, fitted_artifacts=artifacts,
        return_artifacts=True, return_report=True,
    )

    best_row, rows = run_automl(trials, train_df, val_df, full_df, device, logger=logger)
    ranking = rank_results(rows)

    print("\n=== Top-10 trials ===")
    print(ranking.head(10).to_string(index=False))

    _publish_best(
        task, best_row, artifacts, ranking, equipment_id,
        local_task_dir, upload_to_clearml, logger=logger,
    )

    if upload_to_clearml:
        print("Done! Artifacts enviados ao ClearML.")
    else:
        print(f"Done! Artifacts em: {local_task_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML para detecção de anomalias")
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--queue", default="default")
    parser.add_argument("--local-data", action="store_true")
    parser.add_argument("--per-sensor", action="store_true")
    parser.add_argument("--no-clearml-upload", action="store_true")
    parser.add_argument("--local-artifacts-dir", default="artifacts_local")
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=["dense", "lstm", "ocsvm"],
        help="Modelos a incluir (padrão: todos)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Grade reduzida: 20 epochs, 1 preset, 1 threshold — para validar rápido",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()
    main(
        equipment_id=args.equipment,
        remote=args.remote,
        local_data=args.local_data,
        per_sensor=args.per_sensor,
        queue=args.queue,
        upload_to_clearml=not args.no_clearml_upload,
        local_artifacts_dir=args.local_artifacts_dir,
        models=args.models,
        quick=args.quick,
        epochs=args.epochs,
        patience=args.patience,
    )