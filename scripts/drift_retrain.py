"""
drift_triggered_retrain.py
==================================
Walk-forward disparado por DRIFT (não por calendário fixo), testando VÁRIAS
técnicas de detecção de concept drift (as de drift_detectors.py: KS, KS
calibrado, PSI, Page-Hinkley, CUSUM, ADWIN-lite) sobre o mesmo período de dados.

Para cada técnica:
    1. treina um baseline inicial
    2. monitora o erro de reconstrução ponto a ponto com o detector
    3. quando o detector dispara, marca o timestamp e RETREINA o modelo
       usando os dados desde o último retreino como novo baseline (mesmo
       espírito do re-baseline periódico do WalkForwardEvaluator, só que
       disparado por evento em vez de por mês fixo)
    4. recalibra o detector com os erros do novo baseline e continua

Antes de retreinar, cada drift confirmado é classificado (mesma regra do
drift_report.py): tendência monotônica CRESCENTE em vibração/temperatura =
possível degradação → o retreino é BLOQUEADO (retreinar absorveria a falha como
"novo normal") e o alarme segue ativo; sem essa tendência = mudança de
regime/conceito → retreina.

No final, cada técnica é pontuada com as MESMAS funções que o resto do
automl_anomaly_v3.py usa (compute_balanced_score_multi_failure /
compute_balanced_score) — assim os resultados ficam comparáveis lado a lado
com os trials estáticos da busca principal.

Saída, por técnica:
    - retrain_log_<detector>.csv   : um retreino por linha
    - full_scores_<detector>.parquet
    - drift_events_<detector>.csv  : um drift confirmado por linha, com o diagnóstico
                                     degradação física × mudança de regime e a decisão
    - timeline_<detector>.png      : série + linhas verticais nos retreinos
Saída consolidada:
    - resumo_tecnicas.csv          : comparação entre as 7 técnicas
    - timeline_comparativa.png     : as 7 técnicas empilhadas, mesmo eixo X

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
from transpetro_modelos.detector.drift_detectors import build_detector, DETECTOR_NAMES

# reaproveita a lógica de treino/score já existente, sem duplicar
from automl import train_model, score_full, get_true_drift_times

# Piso da idade mínima de uma era (dias): o aquecimento do detector sozinho dá
# só 1-3 dias, o que deixa o retreino disparar com qualquer oscilação.
DEFAULT_MIN_ERA_DAYS = 30


def _kofn(flags: pd.Series, k: int, n: int) -> pd.Series:
    """Persistência k-de-n: só é anomalia se >= k dos últimos n pontos passaram do limiar."""
    return (flags.astype(int).rolling(n, min_periods=n).sum() >= k).astype(bool)


# ════════════════════════════════════════════════════════════════
# Degradação física × mudança de regime (mesma lógica do drift_report.py)
# ════════════════════════════════════════════════════════════════

DEG_PADROES_DEFAULT = ("B-4064A: Vib Mancal Bomba LNA", "B-4064A: Temperatura Mancal Motor LNA")   # sensores físicos onde uma tendência = desgaste
REGIME_COL_DEFAULT = "B-4064A: Corrente"          # variável operacional que define o regime


def sensores_degradacao(colunas, padroes) -> list[str]:
    return [c for c in colunas if any(p.lower() in str(c).lower() for p in padroes)]


def diagnosticar_drift(
    df_pre: pd.DataFrame,
    hist: pd.DataFrame,
    t_fim: pd.Timestamp,
    artifacts=None,
    padroes=DEG_PADROES_DEFAULT,
    regime_col: str = REGIME_COL_DEFAULT,
    janela_dias: int = 30,
    slope_rel_min: float = 1.0,
) -> dict[str, Any]:
    """
    Quando um detector dispara, decide se o desvio parece DEGRADAÇÃO FÍSICA ou
    MUDANÇA DE REGIME/CONCEITO. Mesma regra do drift_report.py:

      - tendência monotônica CRESCENTE em vibração/temperatura (mediana diária
        dos últimos `janela_dias` dias, Theil-Sen)   → "possivel_degradacao"
      - caso contrário                                → "drift_provavel"

    Informativos (não mudam a conclusão): correlação diária erro × variável de
    regime, % de tempo fora da faixa de clip do treino (M5) e nº de episódios de
    anomalia nos últimos 60 dias (intermitente = padrão de drift).

    A tendência é monotônica se a variação total estimada na janela
    (slope x dias) é >= `slope_rel_min` desvios-padrão diários do sensor E a
    fração de dias subindo é > 65% (ou < 35%). É relativa (não em unidade
    física) para funcionar em qualquer sensor; o drift_report.py usa 0,05
    unidade/dia fixo.
    """
    from scipy.stats import theilslopes

    raw = df_pre.loc[:t_fim]
    dia = raw.resample("D").median().dropna(how="all")

    tend: dict[str, dict] = {}
    for c in sensores_degradacao(dia.columns, padroes):
        serie = dia[c].dropna()
        y = serie.tail(janela_dias)
        if len(y) < 15:
            continue
        slope = float(theilslopes(y.values, np.arange(len(y)))[0])
        std_ref = float(serie.std()) or 1e-9
        subindo = float((y.diff().dropna() > 0).mean())
        var_rel = slope * len(y) / std_ref
        monot = abs(var_rel) >= slope_rel_min and (subindo > 0.65 or subindo < 0.35)
        tend[c] = {"slope_dia": slope, "variacao_rel": var_rel,
                   "consistencia": subindo, "monotonica": bool(monot)}
    subindo_fisico = sorted(c for c, v in tend.items() if v["monotonica"] and v["slope_dia"] > 0)

    corr = np.nan
    if regime_col in dia.columns:
        err_d = hist["reconstruction_error"].resample("D").median()
        j = pd.concat([err_d, dia[regime_col]], axis=1, join="inner").dropna()
        if len(j) >= 10:
            corr = float(j.iloc[:, 0].corr(j.iloc[:, 1]))

    fora_clip = np.nan
    cb = getattr(artifacts, "clip_bounds", None)
    if cb:
        ult = raw[raw.index >= t_fim - pd.Timedelta(days=28)]
        fr = [100 * float(((ult[c] < lo) | (ult[c] > hi)).mean())
              for c, (lo, hi) in cb.items() if c in ult.columns]
        fora_clip = max(fr) if fr else np.nan

    al = hist.index[hist["is_anomaly"].to_numpy(dtype=bool)]
    al = al[al >= t_fim - pd.Timedelta(days=60)]
    n_ep = 0 if len(al) == 0 else 1 + int((pd.Series(al).diff() > pd.Timedelta(hours=12)).sum())

    return {
        "conclusao": "possivel_degradacao" if subindo_fisico else "drift_provavel",
        "sensores_subindo": ", ".join(subindo_fisico),
        "n_sensores_avaliados": len(tend),
        "corr_erro_regime": corr,
        "acompanha_regime": bool(abs(corr) > 0.3) if corr == corr else False,
        "fora_clip_pct": fora_clip,
        "n_episodios_60d": n_ep,
        "tendencias": tend,
    }


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
    2x sua sub-janela mínima; o KS calibrado precisa de `k_consecutive`
    janelas acima do limiar para disparar (window_size x k_consecutive).
    Converte esse nº de amostras pra dias usando a taxa de amostragem REAL
    medida nos dados (não assumida).
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
    else:
        # KS calibrado: o disparo exige k janelas acima do limiar (persistência)
        k = getattr(detector_obj, "k_consecutive", None)
        if k is not None:
            warmup_amostras *= k

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
    alarm_sigma: float = 6.5,
    persist_k: int = 15,
    persist_n: int = 20,
    confirm_chunks: int = 2,
    min_train_days: int = 90,
    deg_patterns=DEG_PADROES_DEFAULT,
    regime_col: str = REGIME_COL_DEFAULT,
    deg_window_days: int = 30,
    deg_slope_rel: float = 1.0,
    bloquear_degradacao: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa o walk-forward disparado por drift sobre TODO o df_pre, usando o
    detector `detector_name` (uma chave de DETECTOR_NAMES).

    `min_era_days=None` (default) deriva o valor automaticamente a partir do
    requisito estatístico do próprio detector (auto_min_era_days) — não
    precisa ser informado manualmente.

    Retorna:
        full_scores : DataFrame (reconstruction_error, is_anomaly, era) por timestamp
        retrain_log : DataFrame com um retreino por linha
        drift_events: DataFrame com UMA LINHA POR DRIFT CONFIRMADO (passou no portão),
                      com o diagnóstico degradação × regime e a decisão tomada
                      ("retreinou" ou "retreino_bloqueado_degradacao")
    """
    model_kwargs = model_kwargs or {}
    steps = get_preprocessing_steps(equipment_id, preset=preset)

    inicio = df_pre.index.min()
    fim = df_pre.index.max()
    era_id = 0
    retrain_log: list[dict] = []
    drift_events: list[dict] = []
    full_scores_parts: list[pd.DataFrame] = []

    sens_deg = sensores_degradacao(df_pre.columns, deg_patterns)
    if not sens_deg:
        print(f"  [{detector_name}] [aviso] nenhuma coluna casa com {list(deg_patterns)}: "
              f"sem sensor físico, todo drift será tratado como 'drift_provavel' (nunca bloqueia).")
    if regime_col not in df_pre.columns:
        print(f"  [{detector_name}] [aviso] coluna de regime '{regime_col}' não existe em df_pre "
              f"(a correlação erro × regime ficará vazia).")

    # taxa de amostragem real (amostras/dia): define a janela de 1 dia do KS calibrado
    step_s = df_pre.index.to_series().diff().dt.total_seconds().median() or 300.0
    samples_per_day = max(2, int(round(86400.0 / step_s)))

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
        # Calibra em dado que o modelo não usou para ajustar pesos (os 20% finais).
        # Erro in-sample é sistematicamente menor que em dado novo: limiar e
        # referência do detector saíam apertados demais.
        val_errors = _erro_bruto(model, val_df)
        if len(val_errors) >= 2 * samples_per_day:
            ref_errors = val_errors
        else:
            print(f"  [{detector_name} | {contexto}] [aviso] validação com só {len(val_errors)} "
                  f"amostras (< {2 * samples_per_day}); calibrando com o erro do treino (in-sample).")
            ref_errors = _erro_bruto(model, train_df)

        if alarm_sigma > 0:
            threshold = float(ref_errors.mean() + alarm_sigma * ref_errors.std())
        else:
            threshold = float(np.percentile(ref_errors, threshold_percentile))
        return model, artifacts, ref_errors, threshold

    corte_inicial = inicio + pd.Timedelta(days=initial_train_days)
    train_slice = df_pre.loc[inicio:corte_inicial]
    if len(train_slice) < 50:
        raise ValueError(
            f"[{detector_name}] Janela inicial de {initial_train_days} dias tem só "
            f"{len(train_slice)} amostras — aumenta --initial-train-days."
        )

    model, artifacts, train_errors, threshold = _treinar_era(train_slice, "baseline_inicial")
    # constrói SÓ o detector escolhido (o KS calibrado exige >= 2 janelas de
    # referência e não deve derrubar os demais detectores)
    detector = build_detector(detector_name, train_errors, samples_per_day)

    if min_era_days is None:
        min_era_days = max(auto_min_era_days(detector_name, detector, df_pre), DEFAULT_MIN_ERA_DAYS)
        print(f"[auto] {detector_name}: idade mínima de era = {min_era_days} dia(s) "
              f"(piso de {DEFAULT_MIN_ERA_DAYS})")

    retrain_log.append({
        "detector": detector_name, "era": era_id, "era_start": inicio, "retrain_at": inicio,
        "motivo": "baseline_inicial", "n_amostras_treino": len(train_slice), "threshold": threshold,
    })

    cursor = corte_inicial
    era_start = corte_inicial
    chunks_com_drift = 0   # chunks consecutivos em que o detector disparou

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
        # excedência bruta (ponto a ponto); a persistência k-de-n é aplicada no fim,
        # sobre a série inteira, para atravessar as fronteiras de chunk e de era
        full_scores_parts.append(pd.DataFrame({
            "reconstruction_error": erros, "exceeds": erros > threshold, "era": era_id,
        }, index=pedaco_df.index))

        drift_disparou = False
        for valor in erros:
            if detector.update(float(valor)):
                drift_disparou = True
                break

        # Portão de retreino: o drift precisa (a) persistir por `confirm_chunks`
        # chunks seguidos e (b) a era precisa ter idade mínima. Se o detector
        # disparou mas o portão não abriu, reseta e deixa reacumular: só dispara
        # de novo no chunk seguinte se o drift continuar.
        chunks_com_drift = chunks_com_drift + 1 if drift_disparou else 0
        idade_era_dias = (prox - era_start).days
        if drift_disparou and chunks_com_drift >= confirm_chunks and idade_era_dias >= min_era_days:
            # drift confirmado: antes de retreinar, separa degradação física de mudança de regime
            hist = pd.concat(full_scores_parts).sort_index()
            hist["is_anomaly"] = _kofn(hist["exceeds"], persist_k, persist_n)
            diag = diagnosticar_drift(df_pre, hist, prox, artifacts, deg_patterns,
                                      regime_col, deg_window_days, deg_slope_rel)
            bloquear = bloquear_degradacao and diag["conclusao"] == "possivel_degradacao"
            evento = {"detector": detector_name, "data": prox, "era": era_id,
                      "idade_era_dias": idade_era_dias,
                      **{k: v for k, v in diag.items() if k != "tendencias"},
                      "decisao": "retreino_bloqueado_degradacao" if bloquear else "retreinou"}
            drift_events.append(evento)

            if bloquear:
                print(f"  [{detector_name}] {prox.date()}: drift com tendência crescente em "
                      f"{diag['sensores_subindo']} → POSSÍVEL DEGRADAÇÃO, retreino bloqueado "
                      f"(alarme mantido; avisar a operação).")
                detector.reset()
                chunks_com_drift = 0
            else:
                era_id += 1
                # janela mínima de treino: não treina com uma era curtinha demais
                inicio_treino = max(inicio, min(era_start, prox - pd.Timedelta(days=min_train_days)))
                nova_train_slice = df_pre.loc[inicio_treino:prox]
                model, artifacts, train_errors, threshold = _treinar_era(
                    nova_train_slice, f"retreino_era_{era_id}_em_{prox.date()}"
                )
                detector = build_detector(detector_name, train_errors, samples_per_day)
                retrain_log.append({
                    "detector": detector_name, "era": era_id, "era_start": era_start,
                    "train_start": inicio_treino,
                    "retrain_at": prox, "motivo": f"drift_{detector_name}",
                    "n_amostras_treino": len(nova_train_slice), "threshold": threshold,
                })
                era_start = prox
                chunks_com_drift = 0
        elif drift_disparou:
            detector.reset()

        cursor = prox

    if full_scores_parts:
        full_scores = pd.concat(full_scores_parts).sort_index()
        full_scores["is_anomaly"] = _kofn(full_scores["exceeds"], persist_k, persist_n)
    else:
        full_scores = pd.DataFrame(
            columns=["reconstruction_error", "exceeds", "is_anomaly", "era"]
        )
    return full_scores, pd.DataFrame(retrain_log), pd.DataFrame(drift_events)


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
    eventos: pd.DataFrame | None = None,
) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    if not full_scores.empty:
        ax.plot(full_scores.index, full_scores["reconstruction_error"],
                color="steelblue", linewidth=0.7, alpha=0.8, label="Erro de reconstrução")

        anomalias = full_scores[full_scores["is_anomaly"]]
        if not anomalias.empty:
            ax.scatter(anomalias.index, anomalias["reconstruction_error"],
                      color="black", s=14, zorder=4, edgecolor="none",
                      label=f"Anomalia detectada ({len(anomalias)})")

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

    if eventos is not None and len(eventos):
        bloq = eventos[eventos["decisao"] == "retreino_bloqueado_degradacao"]
        for i, (_, row) in enumerate(bloq.iterrows()):
            ax.axvline(pd.Timestamp(row["data"]), color="darkorange", linestyle="-.", linewidth=1.5,
                      label="Possível degradação (retreino bloqueado)" if i == 0 else None)

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
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["dense", "lstm", "ocsvm", "iforest"],
                        help="Quais modelos testar (default: os 4)")
    parser.add_argument("--presets", nargs="+", default=None,
                        help="Quais presets de preprocessing testar (default: todos os "
                             "disponíveis pro equipamento, via config.preprocess_presets)")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--detectors", nargs="+", default=None,
                        choices=DETECTOR_NAMES,
                        help="Quais técnicas de drift testar (default: todas as 7)")
    parser.add_argument("--initial-train-days", type=int, default=None,
                        help="Dias pro baseline inicial. Default: descoberto automaticamente "
                             "por preset (curva de aprendizado, independente do modelo).")
    parser.add_argument("--min-era-days", type=int, default=None,
                        help="Idade mínima de uma era antes de aceitar retreino. Default: "
                             f"máx(aquecimento do detector, {DEFAULT_MIN_ERA_DAYS} dias).")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--alarm-sigma", type=float, default=6.5,
                        help="Limiar de anomalia = média + N*desvio do erro de validação "
                             "(igual à régua de produção). 0 = usa --threshold-percentile.")
    parser.add_argument("--persist-k", type=int, default=15,
                        help="Persistência: anomalia só se >= K dos últimos N pontos passaram do limiar.")
    parser.add_argument("--persist-n", type=int, default=20)
    parser.add_argument("--confirm-chunks", type=int, default=2,
                        help="Chunks consecutivos com drift exigidos antes de retreinar.")
    parser.add_argument("--min-train-days", type=int, default=90,
                        help="Janela mínima (dias) de treino em cada retreino, mesmo com era curta.")
    parser.add_argument("--deg-patterns", nargs="+", default=list(DEG_PADROES_DEFAULT),
                        help="Trechos dos nomes de coluna dos sensores físicos onde uma tendência "
                             "crescente indica degradação (default: Vibra Temperatura).")
    parser.add_argument("--regime-col", default=REGIME_COL_DEFAULT,
                        help="Coluna de regime operacional p/ correlação com o erro (default: Pressão Descarga).")
    parser.add_argument("--deg-window-days", type=int, default=30,
                        help="Janela (dias) da tendência monotônica (mediana diária, Theil-Sen).")
    parser.add_argument("--deg-slope-rel", type=float, default=1.0,
                        help="Variação mínima na janela, em desvios-padrão diários do sensor, "
                             "para a tendência contar como monotônica.")
    parser.add_argument("--permitir-retreino-em-degradacao", action="store_true",
                        help="Só registra o diagnóstico; NÃO bloqueia o retreino quando há "
                             "possível degradação (útil para comparar com/sem o bloqueio).")
    parser.add_argument("--prefailure-days", type=int, default=30)
    parser.add_argument("--normal-end-days", type=int, default=60)
    parser.add_argument("--output-dir", default="drift_retrain_out")
    parser.add_argument(
        "--max-fp-rate", type=float, default=0.0,
        help="Taxa máxima de FP aceitável (0-1). Combinações acima disso são marcadas "
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
            task_name=f"drift_retrain_{args.equipment}_grid",
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

    # ── resolve a grade: modelos x presets x detectores ──
    model_names = args.models or ["dense", "lstm", "ocsvm", "iforest"]
    available_presets = (
        list(config.preprocess_presets.keys())
        if getattr(config, "preprocess_presets", None) else ["baseline"]
    )
    preset_names = args.presets or available_presets
    detector_names = args.detectors or list(DETECTOR_NAMES)

    total_combos = len(model_names) * len(preset_names) * len(detector_names)
    print(f"\nGrade: {len(model_names)} modelo(s) x {len(preset_names)} preset(s) x "
          f"{len(detector_names)} técnica(s) = {total_combos} combinações")
    print(f"  Modelos:  {model_names}")
    print(f"  Presets:  {preset_names}")
    print(f"  Técnicas: {detector_names}")
    if total_combos > 20:
        print(f"  [aviso] {total_combos} combinações é BASTANTE — cada uma treina o modelo "
              f"do zero a cada retreino disparado. Considere restringir com --models / "
              f"--presets / --detectors se o tempo de execução for um problema.\n")

    # janela inicial: depende só do PREPROCESSING (preset), não do modelo —
    # calcula uma vez por preset e reaproveita entre os modelos
    janela_inicial_por_preset: dict[str, int] = {}
    if args.initial_train_days is not None:
        for p in preset_names:
            janela_inicial_por_preset[p] = args.initial_train_days
        print(f"[manual] usando --initial-train-days {args.initial_train_days} pra todos os presets")
    else:
        for p in preset_names:
            janela_inicial_por_preset[p] = auto_initial_train_days(df_pre, args.equipment, p)

    output_dir = Path(args.output_dir) / f"{args.equipment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_events = getattr(config, "failure_events", None) or (
        [config.failure_date] if getattr(config, "failure_date", None) else None
    )

    resumo_linhas = []
    n_erros = 0

    for model_name in model_names:
        for preset_name in preset_names:
            resultados_combo: dict[str, tuple] = {}
            print(f"\n{'#' * 70}\n# modelo={model_name} | preset={preset_name}\n{'#' * 70}")

            for nome_detector in detector_names:
                print(f"{'=' * 70}\n{model_name} | {preset_name} | {nome_detector}\n{'=' * 70}")
                try:
                    full_scores, retrain_log, drift_events = drift_triggered_walkforward(
                        df_pre, args.equipment, model_name, args.threshold_percentile,
                        nome_detector, device,
                        initial_train_days=janela_inicial_por_preset[preset_name],
                        min_era_days=args.min_era_days,
                        chunk_days=args.chunk_days, preset=preset_name,
                        alarm_sigma=args.alarm_sigma,
                        persist_k=args.persist_k, persist_n=args.persist_n,
                        confirm_chunks=args.confirm_chunks,
                        min_train_days=args.min_train_days,
                        deg_patterns=args.deg_patterns, regime_col=args.regime_col,
                        deg_window_days=args.deg_window_days, deg_slope_rel=args.deg_slope_rel,
                        bloquear_degradacao=not args.permitir_retreino_em_degradacao,
                    )
                except Exception as exc:
                    print(f"  [ERRO] {model_name}|{preset_name}|{nome_detector} falhou: "
                          f"{type(exc).__name__}: {exc}")
                    n_erros += 1
                    continue

                prefixo = f"{model_name}_{preset_name}_{nome_detector}"
                retrain_log.to_csv(output_dir / f"retrain_log_{prefixo}.csv", index=False)
                drift_events.to_csv(output_dir / f"drift_events_{prefixo}.csv", index=False)
                full_scores.to_parquet(output_dir / f"full_scores_{prefixo}.parquet")
                plotar_timeline(
                    full_scores, retrain_log, output_dir / f"timeline_{prefixo}.png",
                    failure_events=failure_events,
                    title=f"{args.equipment} — {model_name}|{preset_name} + {nome_detector}",
                    eventos=drift_events,
                )

                n_retreinos = len(retrain_log) - 1 if len(retrain_log) else 0
                metrics = pontuar_resultado(full_scores, config, args.prefailure_days, args.normal_end_days)
                resumo_linhas.append({
                    "model": model_name, "preset": preset_name, "detector": nome_detector,
                    "n_retreinos": n_retreinos,
                    "n_bloqueios_degradacao": int((drift_events["decisao"] == "retreino_bloqueado_degradacao").sum())
                                              if len(drift_events) else 0,
                    "composite_score": metrics.get("composite_score"),
                    "prefailure_alert_rate": metrics.get("prefailure_alert_rate"),
                    "normal_alert_rate": metrics.get("normal_alert_rate"),
                })
                resultados_combo[nome_detector] = (full_scores, retrain_log)
                print(f"  {n_retreinos} retreino(s) | composite_score={metrics.get('composite_score')}")

            # gráfico comparativo das técnicas, UMA vez por combinação modelo+preset
            if resultados_combo:
                plotar_comparativo(
                    resultados_combo,
                    output_dir / f"timeline_comparativa_{model_name}_{preset_name}.png",
                    failure_events=failure_events,
                    title=f"{args.equipment} — {model_name}|{preset_name}: comparação de técnicas",
                )

    if not resumo_linhas:
        raise RuntimeError("Nenhuma combinação rodou com sucesso.")

    resumo = pd.DataFrame(resumo_linhas)

    if args.max_fp_rate and args.max_fp_rate > 0:
        resumo["aprovado"] = resumo["normal_alert_rate"].fillna(1.0) <= args.max_fp_rate
        n_aprovados = int(resumo["aprovado"].sum())
        print(f"\n[constraint] max_fp_rate={args.max_fp_rate:.2%} → "
              f"{n_aprovados}/{len(resumo)} combinação(ões) dentro do teto")
        resumo = resumo.sort_values(["aprovado", "composite_score"], ascending=[False, False])
    else:
        resumo = resumo.sort_values("composite_score", ascending=False)

    resumo.to_csv(output_dir / "resumo_tecnicas.csv", index=False)
    print(f"\n{'=' * 70}\nRESUMO — TOP 15 COMBINAÇÕES\n{'=' * 70}")
    print(resumo.head(15).to_string(index=False))
    print(f"\n{len(resumo)} combinações no total | {n_erros} falharam | "
          f"resumo completo em resumo_tecnicas.csv")

    print(f"\n✓ Resultados salvos em: {output_dir.resolve()}")

    if task is not None and not args.no_clearml_upload:
        print("\nUpload ao ClearML...")
        task.upload_artifact("resumo_tecnicas", artifact_object=resumo)

        logger = task.get_logger()
        top20 = resumo.head(20)
        for i, row in top20.iterrows():
            rotulo = f"{row['model']}|{row['preset']}|{row['detector']}"
            if row.get("composite_score") is not None:
                logger.report_scalar("drift/composite_score", rotulo,
                                     float(row["composite_score"]), 0)
            logger.report_scalar("drift/n_retreinos", rotulo, float(row["n_retreinos"]), 0)

        # só sobe imagens do TOP 5, pra não estourar o limite de payload do ClearML
        for _, row in resumo.head(5).iterrows():
            prefixo = f"{row['model']}_{row['preset']}_{row['detector']}"
            img = output_dir / f"timeline_{prefixo}.png"
            if img.exists():
                logger.report_image("timelines_top5", prefixo, local_path=str(img), iteration=0)

        logger.report_table("resumo", "grade_completa", table_plot=resumo)

        # UM único artefato ("resultados") com tudo que o notebook de análise precisa:
        # meta, resumo, uma série por combinação, retreinos e eventos de drift.
        # As tabelas vão serializadas em parquet (compacto e independente da versão do pandas).
        import io, json

        def _pq(df: pd.DataFrame) -> bytes:
            buf = io.BytesIO()
            df.to_parquet(buf)
            return buf.getvalue()

        series, logs_all, events_all = {}, [], []
        for _, row in resumo.iterrows():
            prefixo = f"{row['model']}_{row['preset']}_{row['detector']}"
            combo = f"{row['model']} | {row['preset']} | {row['detector']}"
            fs_path = output_dir / f"full_scores_{prefixo}.parquet"
            if fs_path.exists():
                fs = pd.read_parquet(fs_path)
                fs.index.name = "timestamp"
                series[combo] = _pq(fs.reset_index())
            lg_path = output_dir / f"retrain_log_{prefixo}.csv"
            if lg_path.exists():
                logs_all.append(pd.read_csv(lg_path).assign(combo=combo))
            ev_path = output_dir / f"drift_events_{prefixo}.csv"
            if ev_path.exists() and ev_path.stat().st_size > 5:
                events_all.append(pd.read_csv(ev_path).assign(combo=combo))

        pacote = {
            "versao": 1,
            "meta": {
                "equipment": args.equipment,
                "failure_events": [str(e) for e in (failure_events or [])],
                "prefailure_days": args.prefailure_days,
                "normal_end_days": args.normal_end_days,
                "args": json.loads(json.dumps(vars(args), default=str)),
            },
            "resumo": _pq(resumo),
            "series": series,
            "logs": _pq(pd.concat(logs_all, ignore_index=True)) if logs_all else None,
            "eventos": _pq(pd.concat(events_all, ignore_index=True)) if events_all else None,
        }
        # embrulhado em lista para o ClearML serializar como pickle (um dict viraria JSON)
        task.upload_artifact("resultados", artifact_object=[pacote], auto_pickle=True,
                             wait_on_upload=True)
        print(f"✓ Upload completo — artefato único 'resultados' ({len(series)} séries)")


if __name__ == "__main__":
    main()