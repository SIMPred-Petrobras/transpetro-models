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
import signal
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training import checkpoint as ckpt
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
    best_artifacts: Any = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_scores.to_parquet(output_dir / "best_full_scores.parquet")
    with (output_dir / "best_trial.pkl").open("wb") as f:
        pickle.dump({"trial": best_trial, "results": best_row}, f)
    if best_trial.model in ("ocsvm", "isolation_forest", "lof"):
        with (output_dir / "best_model.pkl").open("wb") as f:
            pickle.dump(best_model, f)
    else:
        torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    # PreprocessingArtifacts (scaler/clip_bounds/load_residual_coefs) ajustados no treino.
    # Sem este arquivo não é possível pontuar dados novos no deploy (a normalização seria
    # reajustada na janela nova → escala errada → threshold inválido).
    if best_artifacts is not None:
        with (output_dir / "preprocessing.pkl").open("wb") as f:
            pickle.dump(best_artifacts, f)


def _select_best_row(rows: list[dict], max_fp_rate: float, select_by: str) -> dict | None:
    """Seleciona o melhor row a partir da lista completa (mesma regra usada antes inline):
    FP dentro da constraint E maior detecção pré-falha; desempate por MENOR FP."""
    import numpy as np

    best_key = (-np.inf, -np.inf)
    best = None
    for row in rows:
        fp = float(row["normal_alert_rate"])
        fp_heldout = row.get("val_fp_rate_heldout")
        if select_by == "heldout" and fp_heldout is not None:
            fp_sel = float(fp_heldout)
        else:
            fp_sel = fp
        fp_ok = (max_fp_rate <= 0) or (fp_sel <= max_fp_rate)
        cs = float(row["composite_score"])
        candidate_score = float(row["prefailure_alert_rate"]) if (max_fp_rate > 0) else cs
        candidate_key = (candidate_score, -fp_sel)
        if fp_ok and candidate_key > best_key:
            best_key = candidate_key
            best = row
    return best


def _init_clearml(args: argparse.Namespace, n_trials: int):
    if not (args.clearml or args.remote):
        return None
    if Task is None:
        raise RuntimeError("ClearML não está disponível neste ambiente.")

    # Deve ser chamado antes de Task.init() para ser incluído no hash do venv
    Task.add_requirements("pyarrow")
    Task.add_requirements("torch", package_version="")  # já está na imagem Docker

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or f"automl-anomaly-{args.equipment}",
        output_uri=True,
        reuse_last_task_id=False,
    )
    task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    task.connect({
        "equipment_id": args.equipment,
        "n_trials": n_trials,
        "prefailure_days": args.prefailure_days,
        "normal_end_days": args.normal_end_days,
        "quick": args.quick,
        "models": args.models,
        "presets": args.presets,
        "epochs": args.epochs,
        "patience": args.patience,
    })
    if args.remote:
        task.execute_remotely(queue_name=args.queue)
    return task


def run_grid_search(args: argparse.Namespace):
    import pandas as pd
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = EQUIPMENT_CONFIGS[args.equipment]

    # Resolve janelas de avaliação: CLI explícito > config do equipamento > default global.
    if args.prefailure_days is None:
        args.prefailure_days = config.prefailure_days if config.prefailure_days is not None else 30
    if args.normal_end_days is None:
        args.normal_end_days = config.normal_end_days if config.normal_end_days is not None else 60

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
        latent_dims=_parse_ints(args.latent_dims, []) or None,
        if_n_estimators_list=_parse_ints(args.if_n_estimators, []) or None,
        if_contamination_list=_parse_floats(args.if_contaminations, []) or None,
        lof_n_neighbors_list=_parse_ints(args.lof_n_neighbors, []) or None,
        lof_contamination_list=_parse_floats(args.lof_contaminations, []) or None,
        alarm_policies=args.alarm_policies or None,
        epochs=args.epochs if args.epochs is not None else 100,
        patience=args.patience if args.patience is not None else 10,
        quick=args.quick,
        mode=args.mode,
    )

    task = _init_clearml(args, len(trials))

    print(f"Carregando dados de {args.equipment}...")
    df_raw = load_equipment_data(args.equipment, from_clearml=not args.local_data)
    df_pre, _, _ = run_preprocessing(df_raw, config.pre_split_steps)
    print(f"  Após pré-split: {df_pre.shape}")

    # Guarda contra janela normal vazia: se normal_end cair antes do início dos dados,
    # normal_alert_rate=0 para todos os trials e a constraint --max-fp-rate fica inativa.
    normal_end_ts = pd.Timestamp(config.failure_date) - pd.Timedelta(days=args.normal_end_days)
    n_normal = int((df_pre.index < normal_end_ts).sum())
    print(f"  Janelas de avaliação: prefailure_days={args.prefailure_days}  normal_end_days={args.normal_end_days}")
    if n_normal == 0:
        print(
            f"  [AVISO] Janela normal VAZIA: normal_end={normal_end_ts.date()} é anterior ao "
            f"início dos dados ({df_pre.index.min().date()}). normal_alert_rate=0 para todos os "
            f"trials e a constraint --max-fp-rate NÃO terá efeito. Reduza --normal-end-days."
        )
    else:
        print(f"  Amostras na janela normal (< {normal_end_ts.date()}): {n_normal}")
    print(f"\nTotal de trials: {len(trials)}\n")

    # Constraint hard de falso positivo: só considera "melhor" trial com FP <= max_fp_rate.
    # Valor 0 desativa a constraint (usa composite_score puro como antes).
    max_fp_rate: float = args.max_fp_rate

    # ── checkpoint/resume: semeia rows/attempted a partir do progresso salvo ──
    ckpt_path = Path(args.artifacts_dir) / "progress_checkpoint.pkl" if args.artifacts_dir else None
    rows: list[dict] = []
    attempted: set[str] = set()
    if args.resume:
        payload = ckpt.load_local(ckpt_path) if ckpt_path else None
        origem = "disco local" if payload else ""
        if payload is None and task is not None:
            payload, origem = ckpt.fetch_resume_payload(task, args.resume_from_task)
        if payload:
            rows = list(payload.get("rows", []))
            attempted = set(payload.get("attempted", []))
            print(f"  [RESUME] {len(rows)} trials concluídos, {len(attempted)} tentados "
                  f"(origem: {origem}). Pulando os já feitos.")

    sig_to_trial = {ckpt.trial_signature(t): t for t in trials}

    def _save_checkpoint():
        payload = ckpt.make_payload(rows, attempted)
        if ckpt_path:
            ckpt.save_local(payload, ckpt_path)
        if task is not None:
            try:
                ckpt.upload_to_task(task, payload)
            except Exception as exc:  # nunca derruba o treino por causa do checkpoint
                print(f"  [checkpoint] falha ao subir no ClearML: {exc}")

    # Flush em SIGTERM/SIGINT (botão de abort do ClearML manda SIGTERM antes do kill).
    _flushing = {"done": False}

    def _on_signal(signum, _frame):
        if not _flushing["done"]:
            _flushing["done"] = True
            print(f"\n  [signal {signum}] salvando checkpoint antes de sair...")
            _save_checkpoint()
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
    except ValueError:
        pass  # fora da main thread: sem handler, mas o checkpoint periódico segue ativo

    for i, trial in enumerate(trials, 1):
        sig = ckpt.trial_signature(trial)
        if sig in attempted:
            continue  # já feito num run anterior
        print(f"[{i:03d}/{len(trials):03d}] {trial.label()} ... ", end="", flush=True)
        try:
            row = run_trial(
                trial, args.equipment, df_pre, device,
                prefailure_days=args.prefailure_days,
                normal_end_days=args.normal_end_days,
            )
            attempted.add(sig)  # marca como tentado (pulados/erros não voltam a rodar)
            if row is None:
                print("SKIP (dados insuficientes)")
            else:
                row["_sig"] = sig
                rows.append(row)
                cs = float(row["composite_score"])
                ratio = float(row["discrimination_ratio"])
                fp = float(row["normal_alert_rate"])
                print(
                    f"composite={cs:.4f}  ratio={ratio:.2f}"
                    f"  (pre={float(row['prefailure_alert_rate']):.2%}"
                    f" / normal={fp:.2%})"
                )
                if task is not None:
                    logger = task.get_logger()
                    logger.report_scalar("automl", "composite_score", cs, iteration=i)
                    logger.report_scalar("automl", "discrimination_ratio", ratio, iteration=i)
                    logger.report_scalar("automl", "prefailure_alert_rate", float(row["prefailure_alert_rate"]), iteration=i)
                    logger.report_scalar("automl", "normal_alert_rate", fp, iteration=i)
        except Exception as exc:
            attempted.add(sig)  # erro não-transitório não deve reprocessar indefinidamente
            print(f"ERRO: {exc}")

        if i % args.checkpoint_every == 0:
            _save_checkpoint()

    _save_checkpoint()  # flush final do progresso

    if not rows:
        raise RuntimeError("Nenhum trial válido foi executado.")

    # ── seleção do melhor a partir de TODOS os rows + um único retreino ──
    # (antes o melhor era re-treinado a cada melhora; agora retreina só o vencedor)
    best_model = best_scores = best_artifacts = best_trial = None
    best_row = _select_best_row(rows, max_fp_rate, args.select_by)
    if best_row is not None:
        best_trial = sig_to_trial.get(best_row.get("_sig"))
        if best_trial is None:  # rows de um grid antigo cujos args mudaram
            print("  [AVISO] trial vencedor não está no grid atual (args mudaram?); "
                  "não dá para retreinar/salvar o modelo.")
        else:
            best_model, best_scores, best_artifacts = _retrain_best(
                best_trial, args.equipment, df_pre, device
            )

    results = rank_results(
        rows,
        max_fp_rate=max_fp_rate if max_fp_rate > 0 else None,
        fp_column="val_fp_rate_heldout" if args.select_by == "heldout" else "normal_alert_rate",
    )

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
        _save_best_artifacts(artifacts_dir, best_model, best_scores, best_trial, best_row, best_artifacts)
        print(f"Artefatos do melhor trial salvos em: {artifacts_dir}")

        if task is not None:
            task.upload_artifact("automl_results", artifact_object=results)
            task.upload_artifact("best_full_scores", artifact_object=best_scores)
            trial_dict = asdict(best_trial)
            trial_dict["val_start"] = best_trial.val_start.isoformat() if best_trial.val_start else None
            task.upload_artifact("best_trial", artifact_object={"trial": trial_dict, "results": best_row})
            if best_artifacts is not None:
                # Preprocessing ajustado no treino — necessário para inferência no deploy.
                task.upload_artifact("preprocessing", artifact_object=best_artifacts)
            # Arquivo do modelo — sem isto um run remoto não deixa o modelo recuperável
            # (best_model.pt fica só no disco do worker). Necessário para fetch_and_package.
            model_file = artifacts_dir / (
                "best_model.pkl" if best_trial.model in ("ocsvm", "isolation_forest", "lof")
                else "best_model.pt"
            )
            if model_file.exists():
                task.upload_artifact("best_model", artifact_object=model_file)

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
        val_end_date=config.val_end_date,
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
        latent_dim=trial.latent_dim,
        vae_beta=trial.vae_beta,
        if_n_estimators=trial.if_n_estimators,
        if_contamination=trial.if_contamination,
        lof_n_neighbors=trial.lof_n_neighbors,
        lof_contamination=trial.lof_contamination,
    )
    scores, _, _ = score_full(
        model, trial.model, train_df, full_df,
        trial.threshold_percentile, device,
        seq_len=trial.seq_len,
        batch_size=trial.batch_size,
    )
    # artifacts = scaler/clip_bounds/coefs ajustados no TREINO; precisam ser persistidos
    # para reaplicar o MESMO preprocessing em dados novos no deploy (inferência sem vazamento).
    return model, scores, artifacts


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
    parser.add_argument("--mode", choices=["quick", "full", "extensive"], default="full",
                        help="Modo: quick (5-30 min), full (1 dia), extensive (2 dias) — default: full")
    parser.add_argument("--quick", action="store_true",
                        help="Atalho para --mode quick")
    parser.add_argument("--models", nargs="+",
                        choices=["dense", "lstm", "ocsvm", "vae", "isolation_forest", "lof"], default=None)
    parser.add_argument("--presets", nargs="+", default=None,
                        help="Presets de preprocessing (default: todos disponíveis para o equipamento)")
    parser.add_argument("--thresholds", nargs="+", default=None,
                        help="Percentis do threshold, ex: 90 95 97.5 99 ou '90,95'")
    parser.add_argument("--val-start-dates", nargs="+", default=None,
                        help="Datas de início da validação (YYYY-MM-DD)")
    parser.add_argument("--prefailure-days", type=int, default=None,
                        help="Dias antes da falha que compõem a janela pré-falha "
                             "(default: config.prefailure_days do equipamento, senão 30)")
    parser.add_argument("--normal-end-days", type=int, default=None,
                        help="Dias antes da falha onde termina o período normal "
                             "(default: config.normal_end_days do equipamento, senão 60)")
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
    parser.add_argument("--latent-dims", nargs="+", default=None,
                        help="Dimensões do espaço latente do VAE (default: 8)")
    parser.add_argument("--if-n-estimators", nargs="+", default=None,
                        help="Número de árvores do Isolation Forest (default: 100)")
    parser.add_argument("--if-contaminations", nargs="+", default=None,
                        help="Proporção de outliers esperada no Isolation Forest (default: 0.05)")
    parser.add_argument("--lof-n-neighbors", nargs="+", default=None,
                        help="Número de vizinhos do LOF (default: 20)")
    parser.add_argument("--lof-contaminations", nargs="+", default=None,
                        help="Proporção de outliers esperada no LOF (default: 0.05)")
    parser.add_argument("--remote", action="store_true",
                        help="Envia para execução remota no ClearML (implica --clearml)")
    parser.add_argument("--queue", default="default",
                        help="Fila ClearML para execução remota (default: default)")
    parser.add_argument("--clearml", action="store_true",
                        help="Registra métricas e artefatos no ClearML sem rodar remotamente")
    parser.add_argument("--clearml-project", default="Transpetro")
    parser.add_argument("--clearml-task-name", default=None,
                        help="Nome da task ClearML (default: automl-anomaly-<equipment>)")
    parser.add_argument("--select-by", choices=["insample", "heldout"], default="insample",
                        help="FP usado na seleção/ranking do melhor trial: insample (default, "
                             "normal_alert_rate) ou heldout (val_fp_rate_heldout — FP honesto na "
                             "validação; recomendado). Fallback p/ insample se held-out indisponível.")
    parser.add_argument("--alarm-policies", nargs="+", choices=["threshold", "cusum"], default=None,
                        help="Política(s) de alarme: threshold (percentil, default) e/ou cusum (deriva "
                             "sustentada). Ao usar cusum, restrinja --thresholds (o CUSUM ignora o percentil).")
    parser.add_argument("--max-fp-rate", type=float, default=0.01,
                        help="Constraint máxima de falso positivo no período normal (default: 0.01 = 1%%). "
                             "Use 0 para desabilitar e usar composite_score puro.")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Salva o progresso (trials já feitos) a cada N trials (default: 50). "
                             "Permite retomar uma task que caiu sem recomeçar do zero.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Retoma de um checkpoint anterior se existir (default: ligado).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignora qualquer checkpoint e roda o grid do zero.")
    parser.add_argument("--resume-from-task", default=None,
                        help="ID de uma task ClearML específica de onde retomar o checkpoint "
                             "(override; por padrão usa o nome estável da task).")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.output is None:
        args.output = f"results/automl_{args.equipment}.csv"
    if args.artifacts_dir is None:
        args.artifacts_dir = f"results/automl_{args.equipment}"
    run_grid_search(args)
