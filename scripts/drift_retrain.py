"""
drift_triggered_retrain.py
==================================
Walk-forward disparado por DRIFT (não por calendário fixo), testando VÁRIAS
técnicas de detecção de concept drift (as de drift_detectors.py: KS, PSI,
Page-Hinkley, CUSUM, ADWIN-lite) sobre o mesmo período de dados.

Para cada técnica:
    1. treina um baseline inicial
    2. monitora o erro de reconstrução ponto a ponto com o detector
    3. quando o detector dispara, marca o timestamp e RETREINA o modelo
       usando os dados desde o último retreino como novo baseline (mesmo
       espírito do re-baseline periódico do WalkForwardEvaluator, só que
       disparado por evento em vez de por mês fixo)
    4. recalibra o detector com os erros do novo baseline e continua

No final, cada técnica é pontuada com as MESMAS funções que o resto do
automl_anomaly_v3.py usa (compute_balanced_score_multi_failure /
compute_balanced_score) — assim os resultados ficam comparáveis lado a lado
com os trials estáticos da busca principal.

Saída, por técnica:
    - retrain_log_<detector>.csv   : um retreino por linha
    - full_scores_<detector>.parquet
    - timeline_<detector>.png      : série + linhas verticais nos retreinos
Saída consolidada:
    - resumo_tecnicas.csv          : comparação entre as 6 técnicas
    - timeline_comparativa.png     : as 6 técnicas empilhadas, mesmo eixo X

Uso:
    python scripts/drift_triggered_retrain.py \
        --equipment x --local-data \
        --model dense --threshold-percentile 99 \
        --initial-train-days 60 --min-era-days 14
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training.evaluate import compute_balanced_score, apply_debounce
from transpetro_modelos.training.evaluate_multi_failure import compute_balanced_score_multi_failure
from transpetro_modelos.detector.drift_detectors import default_detectors

# reaproveita a lógica de treino/score já existente, sem duplicar
from automl import train_model, score_full, get_true_drift_times


# ════════════════════════════════════════════════════════════════
# Auto-descoberta: janela inicial e idade mínima de era, a partir dos dados
# ════════════════════════════════════════════════════════════════

def auto_initial_train_days(
    df_pre: pd.DataFrame,
    equipment_id: str,
    preset: str,
    *,
    min_days: int = 7,
    step_days: int = 7,
    cap_days: int = 180,
    holdout_days: int = 7,
    tolerancia: float = 0.03,
) -> int:
    """
    Cresce a janela de baseline em passos de `step_days`, mede o erro de
    reconstrução (via PCA, barato — só pra essa decisão, independente do
    modelo escolhido pro run de verdade) num pedaço de validação logo
    depois de cada candidata. Para de crescer quando adicionar mais dados
    melhora o erro em menos de `tolerancia` (retornos decrescentes) por
    duas candidatas seguidas — ou seja, o próprio dado decide o tamanho.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    steps = get_preprocessing_steps(equipment_id, preset=preset)
    inicio = df_pre.index.min()
    fim = df_pre.index.max()

    print("[auto] descobrindo janela de baseline inicial (curva de aprendizado)...")
    erros_por_candidata = []
    melhoras_pequenas_seguidas = 0
    escolhido = min_days

    dias = min_days
    while dias <= cap_days:
        fim_treino = inicio + pd.Timedelta(days=dias)
        fim_val = fim_treino + pd.Timedelta(days=holdout_days)
        if fim_val > fim:
            break

        treino_raw = df_pre.loc[inicio:fim_treino]
        val_raw = df_pre.loc[fim_treino:fim_val]
        try:
            treino_df, artifacts, _ = run_preprocessing(
                treino_raw, steps, return_artifacts=True, return_report=True
            )
            val_df, _, _ = run_preprocessing(
                val_raw, steps, fitted_artifacts=artifacts,
                return_artifacts=True, return_report=True
            )
        except Exception:
            dias += step_days
            continue

        if len(treino_df) < 50 or len(val_df) < 10:
            dias += step_days
            continue

        pca = PCA(n_components=min(0.95, treino_df.shape[1] - 1) if treino_df.shape[1] > 1 else 1)
        pca.fit(treino_df.values)
        recon = pca.inverse_transform(pca.transform(val_df.values))
        erro_val = float(np.mean((val_df.values - recon) ** 2))
        erros_por_candidata.append((dias, erro_val))
        escolhido = dias

        if len(erros_por_candidata) >= 2:
            anterior = erros_por_candidata[-2][1]
            melhora_relativa = (anterior - erro_val) / anterior if anterior > 0 else 0.0
            print(f"    {dias:>4} dias → erro_val={erro_val:.5f}  (melhora vs. anterior: {melhora_relativa:+.1%})")
            if melhora_relativa < tolerancia:
                melhoras_pequenas_seguidas += 1
                if melhoras_pequenas_seguidas >= 2:
                    break
            else:
                melhoras_pequenas_seguidas = 0
        else:
            print(f"    {dias:>4} dias → erro_val={erro_val:.5f}")

        dias += step_days

    if not erros_por_candidata:
        print(f"[auto] não deu pra medir a curva de aprendizado (dados insuficientes); "
              f"usando o mínimo de {min_days} dias.")
        return min_days

    print(f"[auto] janela de baseline inicial escolhida: {escolhido} dias "
          f"(retornos decrescentes a partir daí)")
    return escolhido


def auto_min_era_days(detector_name: str, detector_obj, df_pre: pd.DataFrame) -> int:
    """
    Deriva a idade mínima de uma era a partir do requisito estatístico
    PRÓPRIO de cada detector (não é um chute): detectores baseados em janela
    (KS/PSI) não têm sinal confiável antes do buffer encher; ADWIN precisa de
    2x sua sub-janela mínima. Converte esse nº de amostras pra dias usando a
    taxa de amostragem REAL medida nos dados (não assumida).
    """
    warmup_amostras = getattr(detector_obj, "window_size", None)
    if warmup_amostras is None:
        min_sub = getattr(detector_obj, "min_subwindow", None)
        if min_sub is not None:
            warmup_amostras = 2 * min_sub
        else:
            # page_hinkley / cusum: não têm buffer formal, mas herdam a mesma
            # exigência de tamanho mínimo de treino que o resto do pipeline já
            # usa (50 amostras) — consistente, não arbitrário.
            warmup_amostras = 50

    step_s = df_pre.index.to_series().diff().dt.total_seconds().median() or 86400.0
    dias = max(1, int(np.ceil(warmup_amostras * step_s / 86400.0)))
    print(f"[auto] {detector_name}: exige {warmup_amostras} amostras de aquecimento "
          f"→ {dias} dia(s) mínimo de era (taxa de amostragem medida: {step_s:.0f}s)")
    return dias


# ════════════════════════════════════════════════════════════════
# Núcleo: walk-forward disparado por drift, para UM detector
# ════════════════════════════════════════════════════════════════

def drift_triggered_walkforward(
    df_pre: pd.DataFrame,
    equipment_id: str,
    model_type: str,
    threshold_percentile: float,
    detector_name: str,
    device: str,
    *,
    initial_train_days: int,
    min_era_days: int | None = None,
    chunk_days: int = 7,
    preset: str = "baseline",
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o walk-forward disparado por drift sobre TODO o df_pre, usando o
    detector `detector_name` (uma chave de default_detectors()).

    `min_era_days=None` (default) deriva o valor automaticamente a partir do
    requisito estatístico do próprio detector (auto_min_era_days) — não
    precisa ser informado manualmente.

    Retorna:
        full_scores : DataFrame (reconstruction_error, is_anomaly, era) por timestamp
        retrain_log : DataFrame com um retreino por linha
    """
    model_kwargs = model_kwargs or {}
    steps = get_preprocessing_steps(equipment_id, preset=preset)

    inicio = df_pre.index.min()
    fim = df_pre.index.max()
    era_id = 0
    retrain_log: list[dict] = []
    full_scores_parts: list[pd.DataFrame] = []

    def _erro_bruto(model, df):
        _, _, train_errors = score_full(
            model, model_type, df, df, threshold_percentile, device
        )
        return train_errors

    def _treinar_era(train_slice: pd.DataFrame, contexto: str = ""):
        train_df, artifacts, _ = run_preprocessing(
            train_slice, steps, return_artifacts=True, return_report=True
        )
        if len(train_df) < 50:
            raise ValueError(
                f"[{detector_name} | {contexto}] Fatia tinha {len(train_slice)} linhas "
                f"antes do preprocessing, {len(train_df)} depois. Período: "
                f"{train_slice.index.min()} → {train_slice.index.max()}. "
                f"Aumenta --initial-train-days / --min-era-days."
            )
        corte = int(len(train_df) * 0.8)
        val_df = train_df.iloc[corte:]
        train_df_fit = train_df.iloc[:corte]
        if len(val_df) < 20:
            val_df = train_df_fit
        if len(train_df_fit) < 20:
            raise ValueError(
                f"[{detector_name} | {contexto}] Sobrou só {len(train_df_fit)} amostras "
                f"de treino após separar val interna."
            )
        model, _ = train_model(model_type, train_df_fit, val_df, device, **model_kwargs)
        train_errors_full = _erro_bruto(model, train_df)
        threshold = float(np.percentile(train_errors_full, threshold_percentile))
        return model, artifacts, train_errors_full, threshold

    corte_inicial = inicio + pd.Timedelta(days=initial_train_days)
    train_slice = df_pre.loc[inicio:corte_inicial]
    if len(train_slice) < 50:
        raise ValueError(
            f"[{detector_name}] Janela inicial de {initial_train_days} dias tem só "
            f"{len(train_slice)} amostras — aumenta --initial-train-days."
        )

    model, artifacts, train_errors, threshold = _treinar_era(train_slice, "baseline_inicial")
    detector = default_detectors(reference=train_errors)[detector_name]

    if min_era_days is None:
        min_era_days = auto_min_era_days(detector_name, detector, df_pre)

    retrain_log.append({
        "detector": detector_name, "era": era_id, "era_start": inicio, "retrain_at": inicio,
        "motivo": "baseline_inicial", "n_amostras_treino": len(train_slice), "threshold": threshold,
    })

    cursor = corte_inicial
    era_start = corte_inicial

    while cursor < fim:
        prox = min(cursor + pd.Timedelta(days=chunk_days), fim)
        pedaco_raw = df_pre.loc[cursor:prox]
        if pedaco_raw.empty:
            cursor = prox
            continue

        try:
            pedaco_df, _, _ = run_preprocessing(
                pedaco_raw, steps, fitted_artifacts=artifacts,
                return_artifacts=True, return_report=True
            )
        except Exception as exc:
            print(f"  [{detector_name}] [aviso] pedaço {cursor.date()} → {prox.date()} "
                  f"falhou no preprocessing ({type(exc).__name__}: {exc}) — pulando.")
            cursor = prox
            continue

        if len(pedaco_df) == 0:
            cursor = prox
            continue

        erros = _erro_bruto(model, pedaco_df)
        is_anomaly = erros > threshold
        full_scores_parts.append(pd.DataFrame({
            "reconstruction_error": erros, "is_anomaly": is_anomaly, "era": era_id,
        }, index=pedaco_df.index))

        drift_disparou = False
        for valor in erros:
            if detector.update(float(valor)):
                drift_disparou = True
                break

        idade_era_dias = (prox - era_start).days
        if drift_disparou and idade_era_dias >= min_era_days:
            era_id += 1
            nova_train_slice = df_pre.loc[era_start:prox]
            model, artifacts, train_errors, threshold = _treinar_era(
                nova_train_slice, f"retreino_era_{era_id}_em_{prox.date()}"
            )
            detector = default_detectors(reference=train_errors)[detector_name]
            retrain_log.append({
                "detector": detector_name, "era": era_id, "era_start": era_start,
                "retrain_at": prox, "motivo": f"drift_{detector_name}",
                "n_amostras_treino": len(nova_train_slice), "threshold": threshold,
            })
            era_start = prox
        elif drift_disparou:
            detector.reset()

        cursor = prox

    full_scores = pd.concat(full_scores_parts).sort_index() if full_scores_parts else pd.DataFrame(
        columns=["reconstruction_error", "is_anomaly", "era"]
    )
    return full_scores, pd.DataFrame(retrain_log)


# ════════════════════════════════════════════════════════════════
# Pontuação: mesmas funções que o resto do pipeline usa
# ════════════════════════════════════════════════════════════════

def pontuar_resultado(
    full_scores: pd.DataFrame,
    config,
    prefailure_days: int,
    normal_end_days: int,
) -> dict[str, Any]:
    """Aplica compute_balanced_score(_multi_failure) na série adaptativa
    inteira — as mesmas contas usadas pros trials estáticos, então o número
    final é diretamente comparável ao resto da busca."""
    failure_events = getattr(config, "failure_events", None)
    failure_date = getattr(config, "failure_date", None)
    scores_df = full_scores[["reconstruction_error", "is_anomaly"]]

    if failure_events:
        is_month_based = all(isinstance(e, str) for e in failure_events)
        return compute_balanced_score_multi_failure(
            scores_df, failure_events=failure_events, prefailure_days=prefailure_days,
            false_positive_penalty=2.0, min_prefailure_rate=0.3,
            aggregation="mean" if not is_month_based else "min",
        )
    if failure_date is not None:
        return compute_balanced_score(
            scores_df, failure_date=failure_date, prefailure_days=prefailure_days,
            normal_end_days=normal_end_days, false_positive_penalty=2.0, min_prefailure_rate=0.3,
        )
    return {"composite_score": None, "prefailure_alert_rate": None, "normal_alert_rate": None}


# ════════════════════════════════════════════════════════════════
# Plot: uma técnica
# ════════════════════════════════════════════════════════════════

def plotar_timeline(
    full_scores: pd.DataFrame,
    retrain_log: pd.DataFrame,
    output_path: Path,
    failure_events: list | None = None,
    title: str = "",
    ax=None,
) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    if not full_scores.empty:
        ax.plot(full_scores.index, full_scores["reconstruction_error"],
                color="steelblue", linewidth=0.7, alpha=0.8, label="Erro de reconstrução")

        for _, row in retrain_log.iterrows():
            era_pontos = full_scores[full_scores["era"] == row["era"]]
            if era_pontos.empty:
                continue
            ax.hlines(row["threshold"], row["retrain_at"], era_pontos.index.max(),
                      color="gray", linestyle=":", linewidth=1,
                      label="Threshold (por era)" if row["era"] == 0 else None)

    retreinos_reais = retrain_log[retrain_log["motivo"] != "baseline_inicial"]
    for i, (_, row) in enumerate(retreinos_reais.iterrows()):
        ax.axvline(row["retrain_at"], color="red", linestyle="--", linewidth=1.3,
                  label="Drift detectado → retreino" if i == 0 else None)

    if failure_events:
        for i, evento in enumerate(failure_events):
            ax.axvline(pd.Timestamp(evento), color="black", linestyle="-", linewidth=1.3,
                      alpha=0.7, label="Falha real" if i == 0 else None)

    ax.set_ylabel("Erro reconstrução", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=True)
    ax.grid(alpha=0.2)

    if standalone:
        ax.set_xlabel("Tempo")
        fig.tight_layout()
        fig.savefig(output_path, dpi=130)
        plt.close(fig)


def plotar_comparativo(
    resultados: dict[str, tuple],  # nome -> (full_scores, retrain_log)
    output_path: Path,
    failure_events: list | None = None,
    title: str = "",
) -> None:
    """Um subplot por técnica, mesmo eixo X, pra comparar visualmente onde
    cada uma decidiu retreinar."""
    n = len(resultados)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(14, 2.6 * n))
    if n == 1:
        axes = [axes]

    for ax, (nome, (full_scores, retrain_log)) in zip(axes, resultados.items()):
        n_retreinos = len(retrain_log) - 1 if len(retrain_log) else 0
        plotar_timeline(full_scores, retrain_log, None, failure_events=failure_events,
                        title=f"{nome} — {n_retreinos} retreino(s)", ax=ax)

    axes[-1].set_xlabel("Tempo")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    parser.add_argument("--local-data", action="store_true")
    parser.add_argument("--model", default="dense", choices=["dense", "lstm", "ocsvm", "iforest"])
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--detectors", nargs="+", default=None,
                        choices=["ks_test", "ks_test_w100", "psi", "page_hinkley", "cusum", "adwin_lite"],
                        help="Quais técnicas testar (default: todas as 6)")
    parser.add_argument("--initial-train-days", type=int, default=None,
                        help="Dias pro baseline inicial. Default: descoberto automaticamente "
                             "via curva de aprendizado (cresce a janela até parar de ajudar).")
    parser.add_argument("--min-era-days", type=int, default=None,
                        help="Idade mínima de uma era antes de aceitar retreino. Default: "
                             "derivado automaticamente do requisito estatístico de cada detector.")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--prefailure-days", type=int, default=30)
    parser.add_argument("--normal-end-days", type=int, default=60)
    parser.add_argument("--output-dir", default="drift_retrain_out")
    parser.add_argument(
        "--max-fp-rate", type=float, default=0.0,
        help="Taxa máxima de FP aceitável (0-1). Técnicas acima disso são marcadas "
             "como reprovadas no resumo. Use 0 para desabilitar."
    )
    # ── ClearML (mesmo padrão do automl_anomaly_v3.py) ──
    parser.add_argument("--remote", action="store_true",
                        help="Envia para execução remota no ClearML e sai")
    parser.add_argument("--queue", default="default")
    parser.add_argument("--clearml-project", default="Transpetro")
    parser.add_argument("--no-clearml-upload", action="store_true",
                        help="Não sobe artefatos ao ClearML")
    parser.add_argument("--no-clearml", action="store_true",
                        help="Não usa ClearML para nada (debug local)")
    args = parser.parse_args()

    # ── inicializa ClearML antes de carregar dados (execute_remotely sai aqui) ──
    task = None
    if not args.no_clearml:
        from clearml import Task
        Task.add_requirements("setuptools>=65.0")
        Task.add_requirements("gitpython>=3.1.40")
        Task.add_requirements("pyarrow")
        Task.add_requirements("torch", package_version="")

        task = Task.init(
            project_name=args.clearml_project,
            task_name=f"drift_retrain_{args.equipment}_{args.model}",
            output_uri=True,
            reuse_last_task_id=False,
        )
        task.connect(vars(args))
        task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")

        if args.remote:
            print(f"Executando remotamente na fila: {args.queue}")
            task.execute_remotely(queue_name=args.queue)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EQUIPMENT_CONFIGS[args.equipment]

    print("Carregando dados...")
    df_raw = load_equipment_data(args.equipment, from_clearml=not args.local_data)
    df_pre, _, _ = run_preprocessing(df_raw, config.pre_split_steps)
    print(f"  Shape: {df_pre.shape} | {df_pre.index.min()} → {df_pre.index.max()}")

    initial_train_days = args.initial_train_days
    if initial_train_days is None:
        initial_train_days = auto_initial_train_days(df_pre, args.equipment, args.preset)
    else:
        print(f"[manual] usando --initial-train-days {initial_train_days} (informado explicitamente)")

    detector_names = args.detectors or list(default_detectors(reference=np.array([0.0, 1.0])).keys())
    print(f"\nTestando {len(detector_names)} técnica(s): {', '.join(detector_names)}\n")

    output_dir = Path(args.output_dir) / f"{args.equipment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_events = getattr(config, "failure_events", None) or (
        [config.failure_date] if getattr(config, "failure_date", None) else None
    )

    resultados_plot: dict[str, tuple] = {}
    resumo_linhas = []

    for nome in detector_names:
        print(f"{'=' * 70}\n{nome}\n{'=' * 70}")
        try:
            full_scores, retrain_log = drift_triggered_walkforward(
                df_pre, args.equipment, args.model, args.threshold_percentile, nome, device,
                initial_train_days=initial_train_days, min_era_days=args.min_era_days,
                chunk_days=args.chunk_days, preset=args.preset,
            )
        except Exception as exc:
            print(f"  [ERRO] {nome} falhou: {type(exc).__name__}: {exc}")
            continue

        retrain_log.to_csv(output_dir / f"retrain_log_{nome}.csv", index=False)
        full_scores.to_parquet(output_dir / f"full_scores_{nome}.parquet")
        plotar_timeline(
            full_scores, retrain_log, output_dir / f"timeline_{nome}.png",
            failure_events=failure_events,
            title=f"{args.equipment} — {args.model} + {nome}",
        )

        n_retreinos = len(retrain_log) - 1 if len(retrain_log) else 0
        metrics = pontuar_resultado(full_scores, config, args.prefailure_days, args.normal_end_days)
        resumo_linhas.append({
            "detector": nome, "n_retreinos": n_retreinos,
            "composite_score": metrics.get("composite_score"),
            "prefailure_alert_rate": metrics.get("prefailure_alert_rate"),
            "normal_alert_rate": metrics.get("normal_alert_rate"),
        })
        resultados_plot[nome] = (full_scores, retrain_log)
        print(f"  {n_retreinos} retreino(s) | composite_score={metrics.get('composite_score')}")

    if not resultados_plot:
        raise RuntimeError("Nenhuma técnica rodou com sucesso.")

    resumo = pd.DataFrame(resumo_linhas)

    # ── constraint de FP: marca quem ficou dentro do teto ──
    if args.max_fp_rate and args.max_fp_rate > 0:
        resumo["aprovado"] = resumo["normal_alert_rate"].fillna(1.0) <= args.max_fp_rate
        n_aprovados = int(resumo["aprovado"].sum())
        print(f"\n[constraint] max_fp_rate={args.max_fp_rate:.2%} → "
              f"{n_aprovados}/{len(resumo)} técnica(s) dentro do teto")
        resumo = resumo.sort_values(["aprovado", "composite_score"], ascending=[False, False])
    else:
        resumo = resumo.sort_values("composite_score", ascending=False)

    resumo.to_csv(output_dir / "resumo_tecnicas.csv", index=False)
    print(f"\n{'=' * 70}\nRESUMO — TÉCNICAS COMPARADAS\n{'=' * 70}")
    print(resumo.to_string(index=False))

    fig_comparativa = plotar_comparativo(
        resultados_plot, output_dir / "timeline_comparativa.png",
        failure_events=failure_events,
        title=f"{args.equipment} — {args.model}: comparação de técnicas de drift",
    )

    print(f"\n✓ Resultados salvos em: {output_dir.resolve()}")

    # ── upload ao ClearML ──
    if task is not None and not args.no_clearml_upload:
        print("\nUpload ao ClearML...")
        task.upload_artifact("resumo_tecnicas", artifact_object=resumo)
        for nome, (full_scores, retrain_log) in resultados_plot.items():
            task.upload_artifact(f"retrain_log_{nome}", artifact_object=retrain_log)
            task.upload_artifact(f"full_scores_{nome}", artifact_object=full_scores)

        logger = task.get_logger()
        for _, row in resumo.iterrows():
            if row.get("composite_score") is not None:
                logger.report_scalar("drift/composite_score", row["detector"],
                                     float(row["composite_score"]), 0)
            logger.report_scalar("drift/n_retreinos", row["detector"],
                                 float(row["n_retreinos"]), 0)

        for nome in resultados_plot:
            img = output_dir / f"timeline_{nome}.png"
            if img.exists():
                logger.report_image("timelines", nome, local_path=str(img), iteration=0)
        comparativa = output_dir / "timeline_comparativa.png"
        if comparativa.exists():
            logger.report_image("timelines", "comparativa",
                                local_path=str(comparativa), iteration=0)

        logger.report_table("resumo", "tecnicas", table_plot=resumo)
        print("✓ Upload completo")


if __name__ == "__main__":
    main()