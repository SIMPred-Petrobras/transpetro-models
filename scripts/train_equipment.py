import argparse
import sys
from pathlib import Path

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


def main(equipment_id: str, remote: bool = False, local_data: bool = False) -> None:
    config = EQUIPMENT_CONFIGS[equipment_id]

    Task.add_requirements("pyarrow")
    task = Task.init(
        project_name="Transpetro",
        task_name=f"autoencoder-{equipment_id}",
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

    # 4. Post-split preprocessing (clip + normalize) — fit on train only, reuse on val/test
    steps = hparams["preprocessing_steps"]
    train_df, scaler, clip_bounds = run_preprocessing(splits["train"], steps, fitted_scaler=None)
    val_df, _, _ = run_preprocessing(splits["val"], steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)
    test_df, _, _ = run_preprocessing(splits["test"], steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)

    n_features = train_df.shape[1]
    encoding_layers = hparams["encoding_layers"]

    # 4. Build model
    model = DenseAutoencoder(input_dim=n_features, encoding_layers=encoding_layers).to(device)
    print(f"  Model input_dim={n_features}, encoding_layers={encoding_layers}")

    # 5. DataLoaders
    train_loader = make_dataloader(train_df, batch_size=hparams["batch_size"], shuffle=True, device=device)
    val_loader = make_dataloader(val_df, batch_size=hparams["batch_size"], shuffle=False, device=device)

    # 6. Train
    print("Training...")
    model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hparams["epochs"],
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        patience=hparams["patience"],
        logger=task.get_logger(),
    )

    # 7. Compute errors and threshold
    train_errors = compute_reconstruction_errors(model, train_df, device=device)
    test_errors = compute_reconstruction_errors(model, test_df, device=device)
    threshold = determine_threshold(train_errors, percentile=hparams["threshold_percentile"])

    n_anomalies = int((test_errors > threshold).sum())
    print(f"  Threshold: {threshold:.6f} | Anomalies in test: {n_anomalies}/{len(test_errors)}")

    # 8. Score test set (with timestamps)
    scores_df = score_test_set(model, test_df, threshold=threshold, device=device)

    # 9. Log scalars to ClearML
    logger = task.get_logger()
    logger.report_scalar("metrics", "threshold", threshold, 0)
    logger.report_scalar("metrics", "train_mse_mean", float(train_errors.mean()), 0)
    logger.report_scalar("metrics", "test_mse_mean", float(test_errors.mean()), 0)
    logger.report_scalar("metrics", "n_anomalies", n_anomalies, 0)

    # 10. Save artifacts
    model_path = f"model_{equipment_id}.pt"
    torch.save(model.state_dict(), model_path)
    task.upload_artifact("model_file", artifact_object=model_path)
    task.upload_artifact("scaler", artifact_object=scaler)
    if clip_bounds:
        task.upload_artifact("clip_bounds", artifact_object=clip_bounds)
    task.upload_artifact("results", artifact_object={
        "threshold": threshold,
        "train_mse_mean": float(train_errors.mean()),
        "train_mse_std": float(train_errors.std()),
        "test_mse_mean": float(test_errors.mean()),
        "test_mse_std": float(test_errors.std()),
        "n_anomalies": n_anomalies,
        "n_test_samples": len(test_errors),
        "n_features": n_features,
        "encoding_layers": encoding_layers,
    })
    task.upload_artifact("test_scores", artifact_object=scores_df)

    # 11. Score full dataset (all periods) for visualization — no influence on training
    print("Scoring full dataset for cross-period analysis...")
    full_df, _, _ = run_preprocessing(df, steps, fitted_scaler=scaler, fitted_clip_bounds=clip_bounds)
    full_scores = score_test_set(model, full_df, threshold=threshold, device=device)
    task.upload_artifact("full_scores", artifact_object=full_scores)
    print(f"  Full dataset scored: {len(full_scores)} samples, {full_scores['is_anomaly'].sum()} anomalies")

    print("Done! Artifacts saved to ClearML.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    parser.add_argument("--remote", action="store_true", help="Submit to ClearML queue for remote execution")
    parser.add_argument("--local-data", action="store_true", help="Load data from local files instead of ClearML")
    args = parser.parse_args()
    main(args.equipment, remote=args.remote, local_data=args.local_data)
