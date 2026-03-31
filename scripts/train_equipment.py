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


def main(equipment_id: str, remote: bool = False) -> None:
    config = EQUIPMENT_CONFIGS[equipment_id]

    task = Task.init(
        project_name="Transpetro",
        task_name=f"autoencoder-{equipment_id}",
        output_uri=True,
    )
    task.set_base_docker("nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04")

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
        "preprocessing_steps": config.preprocessing_steps,
    }
    task.connect(hparams)

    if remote:
        task.execute_remotely(queue_name="default")
    # Everything below runs on the server when remote=True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load data
    print(f"Loading data for {equipment_id}...")
    df = load_equipment_data(equipment_id, from_clearml=True)
    print(f"  Loaded: {df.shape}")

    # 2. Split (before preprocessing to avoid data leakage)
    splits = temporal_split(
        df,
        failure_date=config.failure_date,
        exclusion_days=hparams["exclusion_days"],
    )
    print(f"  Train: {splits['train'].shape}, Val: {splits['val'].shape}, Test: {splits['test'].shape}")

    # 3. Preprocess (fit scaler on train only)
    steps = hparams["preprocessing_steps"]
    train_df, scaler = run_preprocessing(splits["train"], steps, fitted_scaler=None)
    val_df, _ = run_preprocessing(splits["val"], steps, fitted_scaler=scaler)
    test_df, _ = run_preprocessing(splits["test"], steps, fitted_scaler=scaler)

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

    print("Done! Artifacts saved to ClearML.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    parser.add_argument("--remote", action="store_true", help="Submit to ClearML queue for remote execution")
    args = parser.parse_args()
    main(args.equipment, remote=args.remote)
