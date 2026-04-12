import argparse
import pickle
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clearml import Task
from transpetro_modelos.config import EQUIPMENT_CONFIGS
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.data.splitting import temporal_split
from transpetro_modelos.models.autoencoder import DenseAutoencoder
from transpetro_modelos.training.train import train_autoencoder, make_dataloader
from transpetro_modelos.training.evaluate import (
    compute_reconstruction_errors,
    determine_threshold,
    score_test_set,
)


def _sensor_slug(sensor_name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", sensor_name).strip("_").lower()


def _build_sensor_steps(base_steps: list[dict], sensor: str) -> list[dict]:
    return [{"step": "select_features", "features": [sensor]}, *base_steps]


def _save_artifact_local(local_dir: Path, artifact_name: str, artifact_object) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9a-zA-Z_.-]+", "_", artifact_name)

    # If artifact is an existing file path, copy preserving extension.
    if isinstance(artifact_object, (str, Path)):
        src = Path(artifact_object)
        if src.exists() and src.is_file():
            target = local_dir / f"{safe_name}{src.suffix}"
            shutil.copy2(src, target)
            return target

    # DataFrame-friendly storage
    if isinstance(artifact_object, pd.DataFrame):
        target = local_dir / f"{safe_name}.parquet"
        artifact_object.to_parquet(target)
        return target

    # Generic Python object fallback
    target = local_dir / f"{safe_name}.pkl"
    with target.open("wb") as f:
        pickle.dump(artifact_object, f)
    return target


def _publish_artifact(
    task: Task,
    artifact_name: str,
    artifact_object,
    local_dir: Path,
    upload_to_clearml: bool = True,
) -> Path | None:
    local_path = None
    try:
        local_path = _save_artifact_local(local_dir, artifact_name, artifact_object)
    except Exception as e:
        print(f"[WARN] Falha ao salvar artifact local '{artifact_name}': {e}")

    if not upload_to_clearml:
        return local_path

    try:
        task.upload_artifact(artifact_name, artifact_object=artifact_object)
    except Exception as e:
        print(f"[WARN] Falha no upload ClearML do artifact '{artifact_name}': {e}")
        if local_path is not None:
            print(f"[INFO] Artifact '{artifact_name}' preservado localmente em: {local_path}")
    return local_path


def main(
    equipment_id: str,
    remote: bool = False,
    local_data: bool = False,
    per_sensor: bool = False,
    upload_to_clearml: bool = True,
    local_artifacts_dir: str = "artifacts_local",
) -> None:
    config = EQUIPMENT_CONFIGS[equipment_id]

    Task.add_requirements("pyarrow")
    task_name = f"autoencoder-{equipment_id}-per-sensor" if per_sensor else f"autoencoder-{equipment_id}"
    task = Task.init(
        project_name="Transpetro",
        task_name=task_name,
        output_uri=True,
    )
    task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")

    hparams = {
        "equipment_id": equipment_id,
        "encoding_layers": None,  # None = auto based on n_features
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 100,
        "patience": 10,
        "exclusion_days": config.exclusion_days_before,
        "threshold_percentile": 95.0,
        "weight_decay": 1e-5,
        "pre_split_steps": config.pre_split_steps,
        "preprocessing_steps": config.preprocessing_steps,
        "val_start_date": config.val_start_date.isoformat() if config.val_start_date else None,
        "per_sensor_mode": per_sensor,
        "upload_to_clearml": upload_to_clearml,
        "local_artifacts_dir": local_artifacts_dir,
    }
    task.connect(hparams)

    if remote:
        task.execute_remotely(queue_name="default")
    # Everything below runs on the server when remote=True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load data
    print(f"Loading data for {equipment_id}...")
    df = load_equipment_data(equipment_id, from_clearml=not local_data)
    print(f"  Loaded: {df.shape}")

    # 2. Pre-split preprocessing (resample, filter_running) — runs on full dataset
    pre_steps = hparams["pre_split_steps"]
    if pre_steps:
        df, _, _ = run_preprocessing(df, pre_steps, fitted_scaler=None)
        print(f"  After pre-split preprocessing: {df.shape}")

    # 3. Split
    val_start = None
    if hparams["val_start_date"]:
        from datetime import datetime as dt
        val_start = dt.fromisoformat(hparams["val_start_date"])

    splits = temporal_split(
        df,
        failure_date=config.failure_date,
        exclusion_days=hparams["exclusion_days"],
        val_start_date=val_start,
    )
    print(f"  Train: {splits['train'].shape}, Val: {splits['val'].shape}, Test: {splits['test'].shape}")

    logger = task.get_logger()
    base_steps = hparams["preprocessing_steps"]
    local_task_dir = Path(local_artifacts_dir) / f"{task.id}_{equipment_id}"

    if per_sensor:
        sensors = list(df.columns)
        print(f"Per-sensor mode enabled. Training {len(sensors)} models (one per sensor).")
        per_sensor_rows = []

        for sensor in sensors:
            slug = _sensor_slug(sensor)
            steps = _build_sensor_steps(base_steps, sensor)
            print(f"\n=== Sensor: {sensor} ===")

            # Post-split preprocessing (fit on train only, reuse on val/test)
            train_df, scaler, clip_bounds = run_preprocessing(splits["train"], steps, fitted_scaler=None)
            val_df, _, _ = run_preprocessing(
                splits["val"],
                steps,
                fitted_scaler=scaler,
                fitted_clip_bounds=clip_bounds,
            )
            test_df, _, _ = run_preprocessing(
                splits["test"],
                steps,
                fitted_scaler=scaler,
                fitted_clip_bounds=clip_bounds,
            )

            n_features = train_df.shape[1]
            encoding_layers = hparams["encoding_layers"]
            model = DenseAutoencoder(input_dim=n_features, encoding_layers=encoding_layers).to(device)
            print(f"  Model input_dim={n_features}, encoding_layers={encoding_layers}")

            train_loader = make_dataloader(train_df, batch_size=hparams["batch_size"], shuffle=True, device=device)
            val_loader = make_dataloader(val_df, batch_size=hparams["batch_size"], shuffle=False, device=device)

            print("  Training...")
            model = train_autoencoder(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=hparams["epochs"],
                learning_rate=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"],
                patience=hparams["patience"],
                logger=None,
            )

            train_errors = compute_reconstruction_errors(model, train_df, device=device)
            test_errors = compute_reconstruction_errors(model, test_df, device=device)
            threshold = determine_threshold(train_errors, percentile=hparams["threshold_percentile"])
            n_anomalies = int((test_errors > threshold).sum())
            print(f"  Threshold: {threshold:.6f} | Anomalies in test: {n_anomalies}/{len(test_errors)}")

            test_scores = score_test_set(model, test_df, threshold=threshold, device=device)
            full_df, _, _ = run_preprocessing(df, steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)
            full_scores = score_test_set(model, full_df, threshold=threshold, device=device)

            # Per-sensor scalar tracking
            logger.report_scalar("metrics_per_sensor", f"{sensor}/threshold", threshold, 0)
            logger.report_scalar("metrics_per_sensor", f"{sensor}/train_mse_mean", float(train_errors.mean()), 0)
            logger.report_scalar("metrics_per_sensor", f"{sensor}/test_mse_mean", float(test_errors.mean()), 0)
            logger.report_scalar("metrics_per_sensor", f"{sensor}/n_anomalies", n_anomalies, 0)

            model_path = f"model_{equipment_id}__{slug}.pt"
            torch.save(model.state_dict(), model_path)

            results = {
                "sensor": sensor,
                "threshold": threshold,
                "train_mse_mean": float(train_errors.mean()),
                "train_mse_std": float(train_errors.std()),
                "test_mse_mean": float(test_errors.mean()),
                "test_mse_std": float(test_errors.std()),
                "n_anomalies": n_anomalies,
                "n_test_samples": len(test_errors),
                "n_features": n_features,
                "encoding_layers": encoding_layers,
            }

            _publish_artifact(
                task,
                f"model_file__{slug}",
                model_path,
                local_dir=local_task_dir / "per_sensor",
                upload_to_clearml=upload_to_clearml,
            )
            _publish_artifact(
                task,
                f"scaler__{slug}",
                scaler,
                local_dir=local_task_dir / "per_sensor",
                upload_to_clearml=upload_to_clearml,
            )
            if clip_bounds:
                _publish_artifact(
                    task,
                    f"clip_bounds__{slug}",
                    clip_bounds,
                    local_dir=local_task_dir / "per_sensor",
                    upload_to_clearml=upload_to_clearml,
                )
            _publish_artifact(
                task,
                f"results__{slug}",
                results,
                local_dir=local_task_dir / "per_sensor",
                upload_to_clearml=upload_to_clearml,
            )
            _publish_artifact(
                task,
                f"test_scores__{slug}",
                test_scores,
                local_dir=local_task_dir / "per_sensor",
                upload_to_clearml=upload_to_clearml,
            )
            _publish_artifact(
                task,
                f"full_scores__{slug}",
                full_scores,
                local_dir=local_task_dir / "per_sensor",
                upload_to_clearml=upload_to_clearml,
            )

            per_sensor_rows.append(
                {
                    "sensor": sensor,
                    "sensor_slug": slug,
                    "threshold": threshold,
                    "train_mse_mean": float(train_errors.mean()),
                    "test_mse_mean": float(test_errors.mean()),
                    "n_anomalies": n_anomalies,
                    "n_test_samples": len(test_errors),
                }
            )

        summary_df = pd.DataFrame(per_sensor_rows).sort_values("sensor").reset_index(drop=True)
        _publish_artifact(
            task,
            "per_sensor_summary",
            summary_df,
            local_dir=local_task_dir,
            upload_to_clearml=upload_to_clearml,
        )
        print("\nDone! Per-sensor artifacts saved to ClearML.")
        print(summary_df)
        return

    # Default multivariate mode
    steps = base_steps
    train_df, scaler, clip_bounds = run_preprocessing(splits["train"], steps, fitted_scaler=None)
    val_df, _, _ = run_preprocessing(splits["val"], steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)
    test_df, _, _ = run_preprocessing(splits["test"], steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)

    n_features = train_df.shape[1]
    encoding_layers = hparams["encoding_layers"]

    model = DenseAutoencoder(input_dim=n_features, encoding_layers=encoding_layers).to(device)
    print(f"  Model input_dim={n_features}, encoding_layers={encoding_layers}")

    train_loader = make_dataloader(train_df, batch_size=hparams["batch_size"], shuffle=True, device=device)
    val_loader = make_dataloader(val_df, batch_size=hparams["batch_size"], shuffle=False, device=device)

    print("Training...")
    model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hparams["epochs"],
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        patience=hparams["patience"],
        logger=logger,
    )

    train_errors = compute_reconstruction_errors(model, train_df, device=device)
    test_errors = compute_reconstruction_errors(model, test_df, device=device)
    threshold = determine_threshold(train_errors, percentile=hparams["threshold_percentile"])

    n_anomalies = int((test_errors > threshold).sum())
    print(f"  Threshold: {threshold:.6f} | Anomalies in test: {n_anomalies}/{len(test_errors)}")

    scores_df = score_test_set(model, test_df, threshold=threshold, device=device)

    logger.report_scalar("metrics", "threshold", threshold, 0)
    logger.report_scalar("metrics", "train_mse_mean", float(train_errors.mean()), 0)
    logger.report_scalar("metrics", "test_mse_mean", float(test_errors.mean()), 0)
    logger.report_scalar("metrics", "n_anomalies", n_anomalies, 0)

    model_path = f"model_{equipment_id}.pt"
    torch.save(model.state_dict(), model_path)
    _publish_artifact(
        task,
        "model_file",
        model_path,
        local_dir=local_task_dir,
        upload_to_clearml=upload_to_clearml,
    )
    _publish_artifact(
        task,
        "scaler",
        scaler,
        local_dir=local_task_dir,
        upload_to_clearml=upload_to_clearml,
    )
    if clip_bounds:
        _publish_artifact(
            task,
            "clip_bounds",
            clip_bounds,
            local_dir=local_task_dir,
            upload_to_clearml=upload_to_clearml,
        )
    _publish_artifact(
        task,
        "results",
        {
            "threshold": threshold,
            "train_mse_mean": float(train_errors.mean()),
            "train_mse_std": float(train_errors.std()),
            "test_mse_mean": float(test_errors.mean()),
            "test_mse_std": float(test_errors.std()),
            "n_anomalies": n_anomalies,
            "n_test_samples": len(test_errors),
            "n_features": n_features,
            "encoding_layers": encoding_layers,
        },
        local_dir=local_task_dir,
        upload_to_clearml=upload_to_clearml,
    )
    _publish_artifact(
        task,
        "test_scores",
        scores_df,
        local_dir=local_task_dir,
        upload_to_clearml=upload_to_clearml,
    )

    print("Scoring full dataset for cross-period analysis...")
    full_df, _, _ = run_preprocessing(df, steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)
    full_scores = score_test_set(model, full_df, threshold=threshold, device=device)
    _publish_artifact(
        task,
        "full_scores",
        full_scores,
        local_dir=local_task_dir,
        upload_to_clearml=upload_to_clearml,
    )
    print(f"  Full dataset scored: {len(full_scores)} samples, {full_scores['is_anomaly'].sum()} anomalies")

    if upload_to_clearml:
        print("Done! Artifacts enviados ao ClearML (com copia local de seguranca).")
    else:
        print(f"Done! Artifacts salvos localmente em: {local_task_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    parser.add_argument("--remote", action="store_true", help="Submit to ClearML queue for remote execution")
    parser.add_argument("--local-data", action="store_true", help="Load data from local files instead of ClearML")
    parser.add_argument("--per-sensor", action="store_true", help="Train one model per sensor in a single task")
    parser.add_argument(
        "--no-clearml-upload",
        action="store_true",
        help="Nao envia artifacts para o ClearML; salva apenas localmente",
    )
    parser.add_argument(
        "--local-artifacts-dir",
        default="artifacts_local",
        help="Diretorio base para salvar artifacts localmente",
    )
    args = parser.parse_args()
    main(
        args.equipment,
        remote=args.remote,
        local_data=args.local_data,
        per_sensor=args.per_sensor,
        upload_to_clearml=not args.no_clearml_upload,
        local_artifacts_dir=args.local_artifacts_dir,
    )
