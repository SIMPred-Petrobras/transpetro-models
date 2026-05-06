"""
Grid search / AutoML para detecção de anomalias.

Varre combinações de período de treino, preprocessing, modelo, threshold e
hiperparâmetros para qualquer equipamento configurado em EQUIPMENT_CONFIGS.

O ranking usa composite_score = prefailure_alert_rate * (1 - normal_alert_rate).
O discrimination_ratio (prefailure / normal) é incluído como coluna auxiliar.

Uso:
  uv run python scripts/grid_search.py --equipment B-4064A-novos
  uv run python scripts/grid_search.py --equipment B-8802B --quick
  uv run python scripts/grid_search.py --equipment B-4064A-novos --clearml
  uv run python scripts/grid_search.py --equipment B-6511502A --models dense ocsvm --quick
"""

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training.automl import TrialConfig, build_trials, rank_results, run_trial, score_full, train_model

try:
    from clearml import Task
except Exception:
    Task = None


def _parse_dates(values: list[str] | None) -> list[datetime] | None:
    if not values:
        return None
    return [datetime.fromisoformat(v) for v in values]


def _parse_floats(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    result: list[float] = []
    for v in values:
        result.extend(float(item.strip()) for item in v.split(",") if item.strip())
    return result


def _parse_ints(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    result: list[int] = []
    for v in values:
        result.extend(int(item.strip()) for item in v.split(",") if item.strip())
    return result


def _parse_dense_layers(values: list[str] | None) -> list[tuple[int, ...] | None] | None:
    if not values:
        return None
    layers: list[tuple[int, ...] | None] = []
    for v in values:
        if v.lower() in {"auto", "none"}:
            layers.append(None)
        else:
            layers.append(tuple(int(item.strip()) for item in v.split(",") if item.strip()))
    return layers


def _save_best_artifacts(
    output_dir: Path,
    best_model: Any,
    best_scores,
    best_trial: TrialConfig,
    best_row: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_scores.to_parquet(output_dir / "best_full_scores.parquet")
    with (output_dir / "best_trial.pkl").open("wb") as f:
        pickle.dump({"trial": best_trial, "results": best_row}, f)
    if best_trial.model == "ocsvm":
        with (output_dir / "best_model.pkl").open("wb") as f:
            pickle.dump(best_model, f)
    else:
        torch.save(best_model.state_dict(), output_dir / "best_model.pt")


def _init_clearml(args: argparse.Namespace, n_trials: int):
    if not args.clearml:
        return None
    if Task is None:
        raise RuntimeError("ClearML não está disponível neste ambiente.")

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or f"automl-anomaly-{args.equipment}",
        output_uri=True,
        reuse_last_task_id=False,
    )
    task.connect({
        "equipment_id": args.equipment,
        "n_trials": n_trials,
        "prefailure_days": args.prefailure_days,
        "normal_end_days": args.normal_end_days,
        "quick": args.quick,
    })
    return task


def run_grid_search(args: argparse.Namespace):
    import pandas as pd
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = EQUIPMENT_CONFIGS[args.equipment]

    trials = build_trials(
        args.equipment,
        presets=args.presets or None,
        models=args.models or None,
        thresholds=_parse_floats(args.thresholds, []) or None,
        val_start_dates=_parse_dates(args.val_start_dates),
        dense_layers=_parse_dense_layers(args.dense_layers),
        dense_lrs=_parse_floats(args.learning_rates, []) or None,
        batch_sizes=_parse_ints(args.batch_sizes, []) or None,
        seq_lens=_parse_ints(args.seq_lens, []) or None,
        lstm_hidden_dims=_parse_ints(args.lstm_hidden_dims, []) or None,
        lstm_layers=_parse_ints(args.lstm_layers, []) or None,
        ocsvm_nus=_parse_floats(args.ocsvm_nus, []) or None,
        ocsvm_gammas=args.ocsvm_gammas or None,
        epochs=args.epochs if args.epochs is not None else 100,
        patience=args.patience if args.patience is not None else 10,
        quick=args.quick,
    )

    task = _init_clearml(args, len(trials))

    print(f"Carregando dados de {args.equipment}...")
    df_raw = load_equipment_data(args.equipment, from_clearml=not args.local_data)
    df_pre, _, _ = run_preprocessing(df_raw, config.pre_split_steps)
    print(f"  Após pré-split: {df_pre.shape}")
    print(f"\nTotal de trials: {len(trials)}\n")

    rows = []
    best_model = None
    best_scores = None
    best_trial = None
    best_row = None
    best_score = -np.inf

    for i, trial in enumerate(trials, 1):
        print(f"[{i:03d}/{len(trials):03d}] {trial.label()} ... ", end="", flush=True)
        try:
            row = run_trial(
                trial, args.equipment, df_pre, device,
                prefailure_days=args.prefailure_days,
                normal_end_days=args.normal_end_days,
            )
            if row is None:
                print("SKIP (dados insuficientes)")
                continue

            rows.append(row)
            cs = float(row["composite_score"])
            ratio = float(row["discrimination_ratio"])
            print(
                f"composite={cs:.4f}  ratio={ratio:.2f}"
                f"  (pre={float(row['prefailure_alert_rate']):.2%}"
                f" / normal={float(row['normal_alert_rate']):.2%})"
            )

            if task is not None:
                logger = task.get_logger()
                logger.report_scalar("automl", "composite_score", cs, iteration=i)
                logger.report_scalar("automl", "discrimination_ratio", ratio, iteration=i)
                logger.report_scalar("automl", "prefailure_alert_rate", float(row["prefailure_alert_rate"]), iteration=i)
                logger.report_scalar("automl", "normal_alert_rate", float(row["normal_alert_rate"]), iteration=i)

            if cs > best_score:
                best_score = cs
                best_trial = trial
                best_row = row
                # Re-treina para guardar o modelo (run_trial não retorna o modelo)
                best_model, best_scores = _retrain_best(trial, args.equipment, df_pre, device)

        except Exception as exc:
            print(f"ERRO: {exc}")

    if not rows:
        raise RuntimeError("Nenhum trial válido foi executado.")

    results = rank_results(rows)

    print("\n" + "=" * 100)
    print("TOP 10 configurações:")
    print("=" * 100)
    top_cols = [
        "composite_score", "discrimination_ratio",
        "prefailure_alert_rate", "normal_alert_rate",
        "model", "preset", "val_start", "threshold_percentile",
    ]
    print(results[[c for c in top_cols if c in results.columns]].head(10).to_string(index=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nResultados salvos em: {output_path}")

    if (
        args.artifacts_dir
        and best_model is not None
        and best_scores is not None
        and best_trial is not None
        and best_row is not None
    ):
        artifacts_dir = Path(args.artifacts_dir)
        _save_best_artifacts(artifacts_dir, best_model, best_scores, best_trial, best_row)
        print(f"Artefatos do melhor trial salvos em: {artifacts_dir}")

        if task is not None:
            task.upload_artifact("automl_results", artifact_object=results)
            task.upload_artifact("best_full_scores", artifact_object=best_scores)

    if task is not None and best_row is not None:
        logger = task.get_logger()
        for key in ("composite_score", "discrimination_ratio", "normal_alert_rate",
                    "prefailure_alert_rate", "threshold", "n_anomalies"):
            if key in best_row:
                logger.report_scalar("best", key, float(best_row[key]), iteration=0)

    return results


def _retrain_best(trial: TrialConfig, equipment_id: str, df_pre, device: str):
    """Re-treina o melhor trial para obter o modelo e scores finais."""
    import pandas as pd
    from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
    from transpetro_modelos.data.preprocessing import run_preprocessing
    from transpetro_modelos.data.splitting import temporal_split

    config = EQUIPMENT_CONFIGS[equipment_id]
    splits = temporal_split(
        df_pre,
        failure_date=config.failure_date,
        exclusion_days=config.exclusion_days_before,
        val_start_date=trial.val_start,
    )
    steps = get_preprocessing_steps(equipment_id, preset=trial.preset)
    train_df, artifacts, _ = run_preprocessing(splits["train"], steps, return_artifacts=True, return_report=True)
    val_df, _, _ = run_preprocessing(
        splits["val"], steps, fitted_artifacts=artifacts, return_artifacts=True, return_report=True
    )
    full_raw = pd.concat([splits["train"], splits["val"], splits["test"]]).sort_index()
    full_df, _, _ = run_preprocessing(
        full_raw, steps, fitted_artifacts=artifacts, return_artifacts=True, return_report=True
    )

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
    )
    scores, _, _ = score_full(
        model, trial.model, train_df, full_df,
        trial.threshold_percentile, device,
        seq_len=trial.seq_len,
        batch_size=trial.batch_size,
    )
    return model, scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search / AutoML para detecção de anomalias")
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()),
                        help="Equipamento a treinar (chave de EQUIPMENT_CONFIGS)")
    parser.add_argument("--output", default=None,
                        help="Caminho CSV de resultados (default: results/automl_<equipment>.csv)")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Diretório para artefatos do melhor trial (default: results/automl_<equipment>/)")
    parser.add_argument("--local-data", action="store_true",
                        help="Carrega dados locais em vez do ClearML Dataset")
    parser.add_argument("--quick", action="store_true",
                        help="Busca rápida: menos presets, modelos e epochs (smoke test)")
    parser.add_argument("--models", nargs="+", choices=["dense", "lstm", "ocsvm"], default=None)
    parser.add_argument("--presets", nargs="+", default=None,
                        help="Presets de preprocessing (default: todos disponíveis para o equipamento)")
    parser.add_argument("--thresholds", nargs="+", default=None,
                        help="Percentis do threshold, ex: 90 95 97.5 99 ou '90,95'")
    parser.add_argument("--val-start-dates", nargs="+", default=None,
                        help="Datas de início da validação (YYYY-MM-DD)")
    parser.add_argument("--prefailure-days", type=int, default=30,
                        help="Dias antes da falha que compõem a janela pré-falha (default: 30)")
    parser.add_argument("--normal-end-days", type=int, default=60,
                        help="Dias antes da falha onde termina o período normal (default: 60)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rates", nargs="+", default=None)
    parser.add_argument("--batch-sizes", nargs="+", default=None)
    parser.add_argument("--dense-layers", nargs="+", default=None,
                        help="Ex: auto '64,32,16' '128,64,32'")
    parser.add_argument("--seq-lens", nargs="+", default=None)
    parser.add_argument("--lstm-hidden-dims", nargs="+", default=None)
    parser.add_argument("--lstm-layers", nargs="+", default=None)
    parser.add_argument("--ocsvm-nus", nargs="+", default=None)
    parser.add_argument("--ocsvm-gammas", nargs="+", default=None)
    parser.add_argument("--clearml", action="store_true",
                        help="Registra métricas e artefatos no ClearML")
    parser.add_argument("--clearml-project", default="Transpetro")
    parser.add_argument("--clearml-task-name", default=None,
                        help="Nome da task ClearML (default: automl-anomaly-<equipment>)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.output is None:
        args.output = f"results/automl_{args.equipment}.csv"
    if args.artifacts_dir is None:
        args.artifacts_dir = f"results/automl_{args.equipment}"
    run_grid_search(args)
