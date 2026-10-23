"""
drift_triggered_retrain.py
==================================
Simula, sobre o histórico, o CICLO DE PRODUÇÃO do monitoramento: chegam dados
novos (um chunk por vez, default 1 dia) e, a cada chunk, o sistema

    1. PONTUA os dados novos com o modelo vigente (erro de reconstrução);
    2. DETECTA ANOMALIA/FALHA: limiar (média + N·desvio da referência) com
       persistência k-de-n. Isso roda SEMPRE, em paralelo ao drift, e gera os
       episódios de anomalia (alertas_anomalia_*.csv);
    3. DETECTA DRIFT no erro com o detector escolhido (KS, KS calibrado, PSI,
       Page-Hinkley, CUSUM, ADWIN-lite) e só CONFIRMA o drift se ele persistir
       (>= --confirm-disparos disparos espalhados por >= --confirm-dias dias) e
       a era tiver idade mínima;
    4. Drift confirmado -> DIAGNÓSTICO NA JANELA DO REGIME NOVO "falha × regime":
         a) estima QUANDO o regime mudou (CUSUM) e usa a janela dali em diante;
         b) ajusta sensor ~ variáveis de regime (--regime-cols, uma ou várias) na janela
            de treino da era e olha o RESÍDUO do sensor no regime novo (degrau / rampa /
            estável). Se o regime novo está fora da faixa vista no treino (extrapolação),
            o ajuste linear é menos confiável e o limiar do resíduo sobe
            (--extrap-resid-fator);
         c) GATE: bloqueia a adaptação (retreino E recalibração) se
              - um sensor físico mudou além do que o regime explica, ou
              - há anomalia ativa e NENHUMA variável de regime mudou.
            Anomalia ativa COM mudança de regime acompanhada pelos sensores não bloqueia.
            Depois de --teto-bloqueio-dias em bloqueio, o evento é ESCALADO À OPERAÇÃO
            (continua sem adaptar sozinho).
         d) regime novo com < --min-dias-regime-novo dias -> ADIA (sem zerar o detector).
            Com < --min-dias-diagnostico dias não há como separar falha de regime: também
            ADIA (diagnostico_pendente), em vez de bloquear pela tendência de 30 dias.
            Enquanto o MESMO drift segue adiado, o drift_events mantém UMA linha
            (n_chunks_adiado), em vez de uma por dia.
    5. Mudança de conceito -> escolha da AÇÃO, do mais barato ao mais caro:
         - drift LEVE (erro mediano do regime novo <= --limite-leve x a MEDIANA da
           referência): só RECALIBRA a régua (limiar + referência do detector), com as
           anomalias removidas dos erros usados;
         - drift forte: RETREINA a partir do ponto de mudança (sem os episódios
           curtos de anomalia e sem os últimos --quarentena-dias dias) e só ACEITA o
           modelo novo se ele reduzir o erro, medido numa janela que ele NÃO usou no
           treino e em escala relativa à própria referência de cada modelo, em
           >= --min-ganho-retreino vezes; senão REJEITA e mantém o atual.

Critério de "o erro voltou ao normal" (descarta alarme falso e a persistência do ADWIN):
mediana recente <= --fator-nivel-normal x mediana da referência. É a mesma base
(mediana) da decisão de recalibração.

Cada chunk recebe um status (verde / amarelo / vermelho) em chunk_log_*.csv. Anomalia
ativa no chunk deixa o chunk vermelho, independente do detector de drift.

Contaminação: cada retreino/recalibração registra no retrain_log a fração do treino
que estava acima do limiar / em anomalia / em janela de prefalha e a razão do limiar
novo contra o baseline e contra a era anterior (limiares em unidades da mediana da
própria referência, para não depender da escala do scaler de cada era).

Os episódios curtos de anomalia são removidos do treino DEPOIS do preprocessing
(a fatia é pré-processada contínua; assim rolling/lags não costuram datas separadas
por um buraco).

Controle: para cada (modelo, preset) roda também o baseline CONGELADO (sem adaptação)
com a mesma persistência e a mesma pontuação; entra no resumo como detector
"sem_adaptacao", e as demais linhas ganham delta_composite_vs_controle.

O baseline inicial é treinado UMA vez por (modelo, preset) e compartilhado entre
os detectores (comparação justa e bem mais rápido). Com --controle-fim e
--drift-time o script também mede, sobre esse baseline, falsos alarmes e atraso
de detecção de cada detector (drift_benchmark.run_drift_benchmark) e junta no
resumo.

Pontuação final: as MESMAS funções do automl_anomaly_v3.py
(compute_balanced_score_multi_failure / compute_balanced_score). Atenção: a
série pontuada começa depois da janela do baseline inicial.

Saídas por (modelo, preset, detector), em <output-dir>/<equip>_<data>/:
    retrain_log_*.csv      baseline, retreinos e recalibrações (uma por linha) + contaminação
    drift_events_*.csv     um drift confirmado por linha, com diagnóstico e decisão
    chunk_log_*.csv        status (semáforo) de cada chunk
    alertas_anomalia_*.csv episódios de anomalia (início, fim, pico x limiar)
    full_scores_*.parquet  série completa
    timeline_*.png         gráfico (erro, threshold, anomalias, falha real, mudança de conceito)
Consolidadas: resumo_tecnicas.csv (e resumo_parcial.csv durante a execução),
benchmark_<modelo>_<preset>.csv, timeline_comparativa_*.png

Uso:
    python scripts/drift_triggered_retrain.py \
        --equipment x --local-data --no-clearml \
        --models dense --presets baseline \
        --detectors cusum ks_calibrado adwin_lite \
        --regime-cols Corrente \
        --initial-train-days 60 \
        --controle-fim 2025-06-30 --drift-time 2025-09-01 --drift-time 2026-02-01
"""

import argparse
import difflib
import random
import sys
import unicodedata
from dataclasses import dataclass, field, replace
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
from transpetro_modelos.training.evaluate import compute_balanced_score
from transpetro_modelos.training.evaluate_multi_failure import compute_balanced_score_multi_failure
from transpetro_modelos.detector.drift_detectors import build_detector, DETECTOR_NAMES

from automl import train_model, score_full

try:
    from transpetro_modelos.detector.drift_benchmark import run_drift_benchmark
except ImportError:
    try:
        from transpetro_modelos.detector.drift_benchmark import run_drift_benchmark
    except ImportError:
        run_drift_benchmark = None


DEFAULT_MIN_ERA_DAYS = 30
DEFAULT_ALARM_SIGMA = 8.95
DEFAULT_PERSIST_K = 10
DEFAULT_PERSIST_N = 12
DEFAULT_MIN_DIAS_REGIME_NOVO = 21
DEFAULT_BUFFER_TRANSICAO_DIAS = 2
DEFAULT_LIMITE_LEVE = 1.5
DEFAULT_FATOR_NIVEL_NORMAL = 1.1
DEG_PADROES_DEFAULT = ("B-4064A: Vib Mancal Bomba LNA",)
REGIME_COL_DEFAULT = "B-4064A: Corrente"                  
REGIME_COLS_DEFAULT = (REGIME_COL_DEFAULT,) 


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def kofn(flags: pd.Series, k: int, n: int) -> pd.Series:
    return (flags.astype(int).rolling(n, min_periods=n).sum() >= k).astype(bool)


def voltou_ao_normal(erros, ref_errors, fator: float) -> bool:
    e = np.asarray(erros, dtype=float)
    e = e[np.isfinite(e)]
    r = np.asarray(ref_errors, dtype=float)
    r = r[np.isfinite(r)]
    return bool(len(e) > 0 and len(r) > 0 and float(np.median(e)) <= fator * float(np.median(r)))


def _fatia(df: pd.DataFrame, ini, fim, incluir_fim: bool = False) -> pd.DataFrame:
    idx = df.index
    m = idx >= ini
    m &= (idx <= fim) if incluir_fim else (idx < fim)
    return df[m]

def janelas_prefalha(failure_events, prefailure_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    jan = []
    for e in failure_events or ():
        ts = pd.Timestamp(e)
        if isinstance(e, str) and len(e) == 7:
            fim = ts + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        else:
            fim = ts
        jan.append((ts - pd.Timedelta(days=prefailure_days), fim))
    return jan


def em_prefalha(idx, jan) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    m = np.zeros(len(idx), dtype=bool)
    for a, b in jan:
        m |= (idx >= a) & (idx <= b)
    return m


def adaptacoes_em_prefalha(retrain_log: pd.DataFrame, jan) -> int:
    if retrain_log.empty or not jan:
        return 0
    a = retrain_log[retrain_log["motivo"].astype(str) != "baseline_inicial"]
    if a.empty:
        return 0
    return int(em_prefalha(pd.DatetimeIndex(a["retrain_at"]), jan).sum())


def thr_rel(threshold: float, ref_errors) -> float:
    m = float(np.median(np.asarray(ref_errors, dtype=float)))
    return float(threshold / m) if m > 0 else float("nan")


def metricas_contaminacao(idx_treino, hist: pd.DataFrame, thr_rel_novo: float,
                          thr_rel_anterior: float, thr_rel_baseline: float, jan) -> dict:
    idx = pd.DatetimeIndex(idx_treino)
    h = hist.loc[hist.index.isin(idx)]

    def frac(col):
        return float(h[col].astype(float).mean()) if len(h) else float("nan")

    return {
        "treino_frac_exceeds": frac("exceeds"),
        "treino_frac_anomalia": frac("is_anomaly"),
        "treino_cobertura_hist": float(len(h) / len(idx)) if len(idx) else float("nan"),
        "treino_frac_prefalha": float(em_prefalha(idx, jan).mean()) if len(idx) and jan else 0.0,
        "thr_x_baseline": float(thr_rel_novo / thr_rel_baseline) if thr_rel_baseline else float("nan"),
        "thr_x_anterior": float(thr_rel_novo / thr_rel_anterior) if thr_rel_anterior else float("nan"),
    }

def filtrar_anomalias_erros(erros: pd.Series, hist: pd.DataFrame,
                            margem_horas: float) -> tuple[pd.Series, float]:
    an_idx = hist.index[hist["is_anomaly"].to_numpy(dtype=bool)]
    if len(an_idx) == 0 or len(erros) == 0:
        return erros, 0.0
    t = erros.index.values
    a = an_idx.values
    pos = np.searchsorted(a, t)
    esq = a[np.clip(pos - 1, 0, len(a) - 1)]
    dir_ = a[np.clip(pos, 0, len(a) - 1)]
    dist = np.minimum(np.abs(t - esq), np.abs(dir_ - t))
    mask = dist <= np.timedelta64(int(margem_horas * 3600), "s")
    return erros[~mask], float(mask.mean())

def extrair_episodios(
    scores: pd.DataFrame,
    gap_horas: float = 12.0,
    limiar_por_era: dict | None = None,
    coluna: str = "is_anomaly",
) -> pd.DataFrame:
    cols = ["inicio", "fim", "duracao_h", "n_pontos", "era", "pico_erro", "pico_x_limiar"]
    if scores.empty or coluna not in scores:
        return pd.DataFrame(columns=cols)
    an = scores[scores[coluna].to_numpy(dtype=bool)]
    if an.empty:
        return pd.DataFrame(columns=cols)
    ids = (pd.Series(an.index, index=an.index).diff() > pd.Timedelta(hours=gap_horas)).cumsum()
    linhas = []
    for _, g in an.groupby(ids.to_numpy()):
        era = int(g["era"].iloc[0]) if "era" in g else 0
        pico = float(g["reconstruction_error"].max())
        lim = (limiar_por_era or {}).get(era)
        linhas.append({
            "inicio": g.index.min(), "fim": g.index.max(),
            "duracao_h": (g.index.max() - g.index.min()).total_seconds() / 3600.0,
            "n_pontos": len(g), "era": era, "pico_erro": pico,
            "pico_x_limiar": pico / lim if lim else np.nan,
        })
    return pd.DataFrame(linhas, columns=cols)


def intervalos_episodios_curtos(hist: pd.DataFrame, max_dias: float,
                                margem_horas: float) -> list[tuple]:
    ep = extrair_episodios(hist)
    ep = ep[ep["duracao_h"] < max_dias * 24.0]
    m = pd.Timedelta(hours=margem_horas)
    return [(e.inicio - m, e.fim + m) for e in ep.itertuples()]


def excluir_intervalos(df: pd.DataFrame, intervalos) -> tuple[pd.DataFrame, int]:
    if not intervalos or len(df) == 0:
        return df, 0
    mask = np.zeros(len(df), dtype=bool)
    for a, b in intervalos:
        mask |= (df.index >= a) & (df.index <= b)
    if (~mask).mean() < 0.6:
        print("    [aviso] remover os episódios de anomalia deixaria < 60% do treino; mantendo todos.")
        return df, 0
    return df[~mask], int(mask.sum())

def estimar_inicio_regime(
    erros_era: pd.Series,
    ref_errors: np.ndarray,
    k: float = 0.5,
) -> pd.Timestamp:
    if len(erros_era) == 0:
        raise ValueError("estimar_inicio_regime: série da era vazia.")

    ref = np.asarray(ref_errors, dtype=float)
    ref = ref[np.isfinite(ref)]
    eps = 1e-3 * (float(np.median(np.abs(ref))) or 1.0)
    lref = np.log(np.clip(ref, 0.0, None) + eps)
    mu = float(np.median(lref))
    sd = 1.4826 * float(np.median(np.abs(lref - mu)))
    if sd <= 0:
        sd = float(lref.std()) or 1e-6

    vals = np.nan_to_num(erros_era.to_numpy(dtype=float), nan=0.0)
    z = (np.log(np.clip(vals, 0.0, None) + eps) - mu) / sd

    pos, ultimo_zero = 0.0, 0
    for i, v in enumerate(z):
        pos = max(0.0, pos + v - k)
        if pos == 0.0:
            ultimo_zero = i
    return erros_era.index[ultimo_zero]

def _norm_nome(x) -> str:
    return " ".join(unicodedata.normalize("NFC", str(x)).casefold().split())


def reconstruir_nomes(tokens, colunas, exato: bool = True) -> list[str]:
    cols = [str(c) for c in colunas]
    nomes = {_norm_nome(c): c for c in reversed(cols)}
    norm_cols = [_norm_nome(c) for c in cols]

    toks: list[str] = []
    for t in tokens:
        t = str(t)
        if _norm_nome(t) in nomes:
            toks.append(t)
        else:
            toks.extend(x.strip() for x in t.replace("|", ";").split(";") if x.strip())

    def casa(cand: str):
        n = _norm_nome(cand)
        if exato:
            return nomes.get(n)
        return cand if any(n in c for c in norm_cols) else None

    saida, i = [], 0
    while i < len(toks):
        for j in range(len(toks), i, -1):
            achou = casa(" ".join(toks[i:j]))
            if achou is not None:
                saida.append(achou)
                i = j
                break
        else:
            saida.append(toks[i])
            i += 1
    return list(dict.fromkeys(saida))


def sensores_degradacao(colunas, padroes) -> list[str]:
    return [c for c in colunas if any(p.lower() in str(c).lower() for p in padroes)]

def diagnosticar_drift(
    df_pre: pd.DataFrame,
    hist: pd.DataFrame,
    t_fim: pd.Timestamp,
    artifacts=None,
    padroes=DEG_PADROES_DEFAULT,
    regime_cols: tuple = REGIME_COLS_DEFAULT,
    janela_dias: int = 30,
    slope_rel_min: float = 1.0,
) -> dict[str, Any]:
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
    err_d = hist["reconstruction_error"].resample("D").median()
    for rc in regime_cols:
        if rc not in dia.columns:
            continue
        j = pd.concat([err_d, dia[rc]], axis=1, join="inner").dropna()
        if len(j) >= 10:
            c_ = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
            if np.isnan(corr) or abs(c_) > abs(corr):
                corr = c_       

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


def _rob(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    mu = float(np.median(x))
    sd = 1.4826 * float(np.median(np.abs(x - mu)))
    if sd <= 0:
        sd = float(x.std())
    return mu, (sd if sd > 0 else 1e-9)


def avaliar_regime_novo(
    df_raw, ref_ini, ref_fim, novo_ini, novo_fim,
    regime_cols, padroes,
    regime_min: float = 1.0, resid_min: float = 2.0, tend_min: float = 2.0,
    extrap_resid_fator: float = 2.0,
) -> dict | None:
    from scipy.stats import theilslopes

    idx = df_raw.index
    ref = df_raw[(idx >= ref_ini) & (idx < ref_fim)].resample("h").mean()
    novo = df_raw[(idx >= novo_ini) & (idx <= novo_fim)].resample("h").mean()
    if ref.dropna(how="all").shape[0] < 48 or novo.dropna(how="all").shape[0] < 48:
        return None

    reg = [c for c in regime_cols if c in ref.columns and c in novo.columns]

    deltas, extrap_cols = {}, []
    for c in reg:
        mu, sd = _rob(ref[c])
        sd = max(sd, 0.02 * abs(mu), 1e-9)
        med_novo = float(novo[c].median())
        deltas[c] = (med_novo - mu) / sd
        lo, hi = np.nanpercentile(ref[c], [1, 99])
        if not (lo <= med_novo <= hi):
            extrap_cols.append(c)
    mudaram = sorted(c for c, d in deltas.items() if np.isfinite(d) and abs(d) >= regime_min)
    regime_mudou = bool(mudaram)
    extrapolacao = bool(extrap_cols)
    delta_max = max(deltas.values(), key=abs) if deltas else float("nan")

    resid_lim = resid_min * (extrap_resid_fator if extrapolacao else 1.0)

    detalhes: dict[str, dict] = {}
    for c in ref.columns:
        if c in reg or c not in novo.columns: 
            continue
        if not any(p.lower() in str(c).lower() for p in padroes):
            continue
        r, n = ref[[c] + reg].dropna(), novo[[c] + reg].dropna()
        if len(r) < 48 or len(n) < 24:
            continue
        usa = [x for x in reg if float(r[x].std()) > 0]
        if usa:
            Xr = np.column_stack([np.ones(len(r))] + [r[x].to_numpy() for x in usa])
            Xn = np.column_stack([np.ones(len(n))] + [n[x].to_numpy() for x in usa])
            coef, *_ = np.linalg.lstsq(Xr, r[c].to_numpy(), rcond=None)
            res_ref = r[c].to_numpy() - Xr @ coef
            res_novo = pd.Series(n[c].to_numpy() - Xn @ coef, index=n.index)
        else:
            a = float(r[c].median())
            res_ref = r[c].to_numpy() - a
            res_novo = n[c] - a
        c0, sd_res = _rob(res_ref)
        resid_med = float((np.median(res_novo.to_numpy()) - c0) / sd_res)

        rd = res_novo.resample("D")
        d = rd.median()[rd.count() >= 18].dropna()
        tend = 0.0
        if len(d) >= 7:
            slope = float(theilslopes(d.to_numpy(), np.arange(len(d)))[0])
            tend = slope * len(d) / sd_res
        if resid_med >= resid_lim:
            forma = "rampa" if tend >= tend_min else "degrau"
        elif tend >= tend_min:
            forma = "rampa_incipiente"
        else:
            forma = "estavel"
        detalhes[c] = {"resid_sd": resid_med, "tend_sd": float(tend), "forma": forma,
                       "fora": bool(resid_med >= resid_lim or tend >= tend_min)}

    fora = sorted(c for c, v in detalhes.items() if v["fora"])
    cand = fora or list(detalhes)
    pior = max(cand, key=lambda c: detalhes[c]["resid_sd"]) if cand else None
    return {
        "delta_regime_sd": float(delta_max),
        "deltas_regime": deltas,
        "regimes_mudaram": mudaram,
        "regime_mudou": regime_mudou,
        "sensores_fora": fora,
        "resid_max_sd": float(max((v["resid_sd"] for v in detalhes.values()), default=float("nan"))),
        "tend_max_sd": float(max((v["tend_sd"] for v in detalhes.values()), default=float("nan"))),
        "forma": detalhes[pior]["forma"] if pior else "sem_sensor",
        "explicada_por_regime": bool(regime_mudou and not fora),
        "extrapolacao": extrapolacao,
        "resid_lim_usado": float(resid_lim),
        "n_sensores": len(detalhes),
        "detalhes": detalhes,
    }

MIN_DIAS_RECALIBRACAO = 2


def decidir_recalibracao(
    erros_novos,
    ref_errors,
    alarm_sigma: float,
    threshold_percentile: float,
    samples_per_day: int,
    limite_leve: float = DEFAULT_LIMITE_LEVE,
    winsor_percentil: float = 99.5,
    robusto: bool = False,
    n_mad: float = 6.0,
) -> dict:
    novos = np.asarray(erros_novos, dtype=float)
    novos = novos[np.isfinite(novos)]
    ref = np.asarray(ref_errors, dtype=float)
    ref = ref[np.isfinite(ref)]
    vazio = {"acao": "retreinar", "erro_rel": float("nan"),
             "threshold": None, "ref_errors": None, "motivo": ""}

    if len(novos) < MIN_DIAS_RECALIBRACAO * samples_per_day:
        return {**vazio, "motivo": f"só {len(novos)} amostras do regime novo"}
    med_ref = float(np.median(ref))
    if med_ref <= 0:
        return {**vazio, "motivo": "mediana da referência não positiva"}

    erro_rel = float(np.median(novos)) / med_ref
    if erro_rel > limite_leve:
        return {**vazio, "erro_rel": erro_rel,
                "motivo": f"erro relativo {erro_rel:.2f}x > {limite_leve}x"}

    ref_nova = np.clip(novos, None, np.percentile(novos, winsor_percentil))
    if robusto:
        thr = limiar_robusto(ref_nova, n_mad)
    elif alarm_sigma > 0:
        thr = float(ref_nova.mean() + alarm_sigma * ref_nova.std())
    else:
        thr = float(np.percentile(ref_nova, threshold_percentile))
    return {"acao": "recalibrar", "erro_rel": erro_rel, "threshold": thr,
            "ref_errors": ref_nova, "motivo": f"erro relativo {erro_rel:.2f}x <= {limite_leve}x"}


@dataclass
class Era:
    model: Any
    artifacts: Any
    ref_errors: np.ndarray  
    threshold: float
    val_index: pd.Index   
    val_errors: np.ndarray 
    train_ini: Any = None 
    train_fim: Any = None
    train_index: Any = None
    n_excluidas: int = 0


@dataclass
class Params:
    initial_train_days: int
    min_era_days: int | None = None
    chunk_days: int = 1
    lookback_horas: float = 6.0
    alarm_sigma: float = DEFAULT_ALARM_SIGMA
    persist_k: int = DEFAULT_PERSIST_K
    persist_n: int = DEFAULT_PERSIST_N
    confirm_dias: float = 3.0
    confirm_disparos: int = 2
    expira_suspeita_dias: float = 10.0
    min_dias_regime_novo: int = DEFAULT_MIN_DIAS_REGIME_NOVO
    buffer_transicao_dias: int = DEFAULT_BUFFER_TRANSICAO_DIAS
    fator_nivel_normal: float = DEFAULT_FATOR_NIVEL_NORMAL
    deg_patterns: tuple = DEG_PADROES_DEFAULT
    regime_cols: tuple = REGIME_COLS_DEFAULT
    deg_window_days: int = 30
    deg_slope_rel: float = 1.0
    bloquear_degradacao: bool = True
    cooldown_bloqueio_dias: int = 14
    permitir_recalibracao: bool = True
    limite_leve: float = DEFAULT_LIMITE_LEVE
    min_ganho_retreino: float = 1.2
    cooldown_rejeicao_dias: int = 7
    excluir_anomalias_treino: bool = True
    max_dias_episodio_excluir: float = 3.0
    margem_anomalia_horas: float = 24.0
    min_horas_politica: float = 4000.0
    seed: int = 0
    model_kwargs: dict = field(default_factory=dict)
    failure_events: tuple = ()
    prefailure_days: int = 30
    gate_anomalia: bool = True
    gate_janela_dias: int = 7
    gate_exceeds_pct: float = 3.0      
    teto_bloqueio_dias: int = 45    
    max_frac_filtrada_recal: float = 0.4
    quarentena_dias: int = 2
    min_dias_diagnostico: int = 7
    regime_delta_min: float = 1.0
    resid_min: float = 2.0
    tend_min: float = 2.0
    extrap_resid_fator: float = 2.0
    gate_continuo: bool = True
    persist_contexto_n: int = 2 
    limiar_robusto: bool = False
    limiar_mad: float = 6.0


def erros_modelo(model, model_type: str, df: pd.DataFrame, threshold_percentile: float,
                 device: str) -> pd.Series:
    _, _, e = score_full(model, model_type, df, df, threshold_percentile, device)
    e = np.asarray(e, dtype=float)
    return pd.Series(e, index=df.index[-len(e):])


def ganho_relativo(df_pre: pd.DataFrame, steps, era_antiga: Era, era_nova: Era,
                   model_type: str, threshold_percentile: float, device: str,
                   ini, fim, lookback_horas: float, min_amostras: int) -> float:

    def rel(era: Era):
        raw = _fatia(df_pre, ini - pd.Timedelta(hours=lookback_horas), fim)
        pre, _, _ = run_preprocessing(raw, steps, fitted_artifacts=era.artifacts,
                                      return_artifacts=True, return_report=True)
        pre = pre[pre.index >= ini]
        if len(pre) < min_amostras:
            return None
        e = erros_modelo(era.model, model_type, pre, threshold_percentile, device)
        return float(np.median(e)) / max(float(np.median(era.ref_errors)), 1e-12)

    a, n = rel(era_antiga), rel(era_nova)
    if a is None or n is None:
        return float("nan")
    return a / max(n, 1e-12)


def treinar_era(
    df_slice: pd.DataFrame,
    steps,
    model_type: str,
    device: str,
    threshold_percentile: float,
    alarm_sigma: float,
    samples_per_day: int,
    seed: int,
    model_kwargs: dict,
    rotulo: str,
    excluir: list | None = None,
    robusto: bool = False,
    n_mad: float = 6.0,
) -> Era:
    train_df, artifacts, _ = run_preprocessing(
        df_slice, steps, return_artifacts=True, return_report=True
    )
    train_df, n_excl = excluir_intervalos(train_df, excluir)
    if len(train_df) < 50:
        raise ValueError(
            f"[{rotulo}] Fatia tinha {len(df_slice)} linhas antes do preprocessing, "
            f"{len(train_df)} depois ({df_slice.index.min()} → {df_slice.index.max()})."
        )
    corte = int(len(train_df) * 0.8)
    val_df = train_df.iloc[corte:]
    train_fit = train_df.iloc[:corte]
    if len(val_df) < 20:
        val_df = train_fit
    if len(train_fit) < 20:
        raise ValueError(f"[{rotulo}] Sobrou só {len(train_fit)} amostras de treino após separar a validação.")

    set_seed(seed)
    model, _ = train_model(model_type, train_fit, val_df, device, **model_kwargs)
    val_errors = erros_modelo(model, model_type, val_df, threshold_percentile, device).to_numpy()
    if len(val_errors) >= 2 * samples_per_day:
        ref_errors = val_errors
    else:
        print(f"    [{rotulo}] [aviso] validação com só {len(val_errors)} amostras "
              f"(< {2 * samples_per_day}); calibrando com o erro do treino (in-sample).")
        ref_errors = erros_modelo(model, model_type, train_df, threshold_percentile, device).to_numpy()

    if robusto:
        threshold = limiar_robusto(ref_errors, n_mad)
    elif alarm_sigma > 0:
        threshold = float(ref_errors.mean() + alarm_sigma * ref_errors.std())
    else:
        threshold = float(np.percentile(ref_errors, threshold_percentile))
    return Era(model, artifacts, ref_errors, threshold, val_df.index, val_errors,
               train_ini=df_slice.index.min(),
               train_fim=df_slice.index.max() + pd.Timedelta(seconds=1),
               train_index=train_df.index, n_excluidas=n_excl)


def samples_per_day_de(df: pd.DataFrame) -> int:
    step_s = df.index.to_series().diff().dt.total_seconds().median() or 300.0
    return max(2, int(round(86400.0 / step_s)))


def treinar_baseline(df_pre, equipment_id, model_type, preset, device,
                     threshold_percentile, p: Params, spd: int) -> Era:
    steps = get_preprocessing_steps(equipment_id, preset=preset)
    inicio = df_pre.index.min()
    corte = inicio + pd.Timedelta(days=p.initial_train_days)
    fatia = _fatia(df_pre, inicio, corte)
    if len(fatia) < 50:
        raise ValueError(f"Janela inicial de {p.initial_train_days} dias tem só {len(fatia)} "
                         f"amostras — aumente --initial-train-days.")
    return treinar_era(fatia, steps, model_type, device, threshold_percentile, p.alarm_sigma,
                       spd, p.seed, p.model_kwargs, f"baseline_{model_type}_{preset}",
                       robusto=p.limiar_robusto, n_mad=p.limiar_mad)

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

    from sklearn.decomposition import PCA

    steps = get_preprocessing_steps(equipment_id, preset=preset)
    inicio, fim = df_pre.index.min(), df_pre.index.max()
    cap_eff = min(cap_days, (fim - inicio).days - holdout_days - 1)
    print("[auto] descobrindo janela de baseline inicial (curva de aprendizado)...")
    if cap_eff < min_days:
        print(f"[auto] dados insuficientes para a curva; usando o mínimo de {min_days} dias.")
        return min_days

    val_ini = inicio + pd.Timedelta(days=cap_eff)
    val_raw = _fatia(df_pre, val_ini, val_ini + pd.Timedelta(days=holdout_days))

    medidas: list[tuple[int, float]] = []
    pequenas_seguidas, plato = 0, None
    dias = min_days
    while dias <= cap_eff:
        treino_raw = _fatia(df_pre, inicio, inicio + pd.Timedelta(days=dias))
        try:
            treino_df, artifacts, _ = run_preprocessing(
                treino_raw, steps, return_artifacts=True, return_report=True)
            val_df, _, _ = run_preprocessing(
                val_raw, steps, fitted_artifacts=artifacts,
                return_artifacts=True, return_report=True)
        except Exception:
            dias += step_days
            continue
        if len(treino_df) < 50 or len(val_df) < 10:
            dias += step_days
            continue

        pca = PCA(n_components=0.95 if treino_df.shape[1] > 1 else 1)
        pca.fit(treino_df.values)
        recon = pca.inverse_transform(pca.transform(val_df.values))
        erro_val = float(np.mean((val_df.values - recon) ** 2))
        medidas.append((dias, erro_val))

        if len(medidas) >= 2:
            anterior = medidas[-2][1]
            melhora = (anterior - erro_val) / anterior if anterior > 0 else 0.0
            print(f"    {dias:>4} dias → erro_val={erro_val:.5f}  (melhora vs. anterior: {melhora:+.1%})")
            if melhora < tolerancia:
                if pequenas_seguidas == 0:
                    plato = medidas[-2][0]   # o candidato anterior já bastava
                pequenas_seguidas += 1
                if pequenas_seguidas >= 2:
                    break
            else:
                pequenas_seguidas, plato = 0, None
        else:
            print(f"    {dias:>4} dias → erro_val={erro_val:.5f}")
        dias += step_days

    if not medidas:
        print(f"[auto] não deu para medir a curva; usando o mínimo de {min_days} dias.")
        return min_days
    escolhido = plato if plato else medidas[-1][0]
    print(f"[auto] janela de baseline inicial escolhida: {escolhido} dias")
    return escolhido


def auto_min_era_days(detector_name: str, detector_obj, df_pre: pd.DataFrame) -> int:
    warmup = getattr(detector_obj, "window_size", None)
    if warmup is None:
        min_sub = getattr(detector_obj, "min_subwindow", None)
        warmup = 2 * min_sub if min_sub is not None else 50
    else:
        k = getattr(detector_obj, "k_consecutive", None)
        if k is not None:
            warmup *= k
    step_s = df_pre.index.to_series().diff().dt.total_seconds().median() or 86400.0
    dias = max(1, int(np.ceil(warmup * step_s / 86400.0)))
    print(f"[auto] {detector_name}: aquecimento de {warmup} amostras → {dias} dia(s)")
    return dias

def benchmark_detectores(
    df_pre, equipment_id, model_type, preset, device, threshold_percentile,
    baseline: Era, p: Params, spd: int, nomes: list[str],
    controle_fim: pd.Timestamp, drift_times: list[pd.Timestamp], horizonte_dias: int,
) -> pd.DataFrame:
    if run_drift_benchmark is None:
        print("[aviso] drift_benchmark não encontrado: benchmark pulado.")
        return pd.DataFrame()
    steps = get_preprocessing_steps(equipment_id, preset=preset)
    corte = df_pre.index.min() + pd.Timedelta(days=p.initial_train_days)
    resto, _, _ = run_preprocessing(
        df_pre[df_pre.index >= corte], steps, fitted_artifacts=baseline.artifacts,
        return_artifacts=True, return_report=True)
    serie = erros_modelo(baseline.model, model_type, resto, threshold_percentile, device)

    controle = serie.loc[:controle_fim]
    if controle.empty:
        print("[aviso] controle vazio (--controle-fim antes do fim do baseline): benchmark pulado.")
        return pd.DataFrame()
    dias_controle = max(1.0, (controle.index.max() - controle.index.min()).total_seconds() / 86400.0)
    drift_times = sorted(t for t in drift_times if t > controle_fim)

    linhas = []
    for nome in nomes:
        try:
            det_c = build_detector(nome, baseline.ref_errors, spd)
        except ValueError as exc:
            print(f"  [benchmark] {nome} ignorado: {exc}")
            continue
        r_c = run_drift_benchmark(controle, {nome: det_c}, pd.Timestamp("2100-01-01")).iloc[0]
        falsos = int(r_c["n_false_alarms_before_drift"])

        atrasos_d, detectados = [], 0
        anterior = controle_fim
        for t in drift_times:
            trecho = serie.loc[anterior:t + pd.Timedelta(days=horizonte_dias)]
            r = run_drift_benchmark(
                trecho, {nome: build_detector(nome, baseline.ref_errors, spd)}, t).iloc[0]
            if r["detected"]:
                detectados += 1
                atrasos_d.append(r["detection_delay_time"].total_seconds() / 86400.0)
            anterior = t + pd.Timedelta(days=horizonte_dias)
        linhas.append({
            "detector": nome,
            "bench_falsos_por_100d": round(100.0 * falsos / dias_controle, 2),
            "bench_taxa_deteccao": round(detectados / len(drift_times), 2) if drift_times else np.nan,
            "bench_atraso_mediano_dias": round(float(np.median(atrasos_d)), 2) if atrasos_d else np.nan,
        })
    return pd.DataFrame(linhas)

def _mesclar_adiamento(drift_events: list[dict]) -> None:
    if len(drift_events) < 2:
        return
    atual, ant = drift_events[-1], drift_events[-2]
    if (str(atual["decisao"]).startswith("retreino_adiado")
            and str(ant["decisao"]).startswith("retreino_adiado")
            and ant["era"] == atual["era"]
            and ant["primeiro_disparo"] == atual["primeiro_disparo"]):
        n = ant.get("n_chunks_adiado", 1) + 1
        primeira = ant.get("primeira_avaliacao", ant["data"])
        ant.update(atual)
        ant["n_chunks_adiado"], ant["primeira_avaliacao"] = n, primeira
        drift_events.pop()

CONTEXTOS = ("normal", "regime_novo", "falha_sensor", "anomalia_sem_regime", "indeterminado")
CONTEXTOS_FALHA = ("falha_sensor", "anomalia_sem_regime")


def classificar_contexto(
    avaliar_regime_novo,
    df_pre: pd.DataFrame,
    era,
    hist_exceeds: pd.Series,
    t_fim: pd.Timestamp,
    p,
    mascara_operacao: pd.Series | None = None,
) -> dict:
    ini = t_fim - pd.Timedelta(days=p.gate_janela_dias)
    rec = hist_exceeds[hist_exceeds.index >= ini]
    exc_pct = 100.0 * float(rec.mean()) if len(rec) else 0.0
    out = {"contexto": "indeterminado", "exc_pct_janela": exc_pct,
           "regimes_mudaram": "", "sensores_fora": "", "forma": "-"}

    if era.train_ini is None:
        return out
    df_gate = df_pre if mascara_operacao is None else df_pre[mascara_operacao.to_numpy()]
    av = avaliar_regime_novo(
        df_gate, era.train_ini, era.train_fim, ini, t_fim,
        p.regime_cols, p.deg_patterns, p.regime_delta_min,
        p.resid_min, p.tend_min, p.extrap_resid_fator)
    if av is None:
        return out

    out.update(regimes_mudaram=", ".join(av["regimes_mudaram"]),
               sensores_fora=", ".join(av["sensores_fora"]), forma=av["forma"])
    if av["sensores_fora"]:
        out["contexto"] = "falha_sensor"
    elif av["regime_mudou"]:
        out["contexto"] = "regime_novo"
    elif exc_pct >= p.gate_exceeds_pct:
        out["contexto"] = "anomalia_sem_regime"
    else:
        out["contexto"] = "normal"
    return out


def marcar_operacao(mascara: pd.Series, raw_chunk_idx: pd.DatetimeIndex,
                    pre_idx: pd.DatetimeIndex) -> None:
    if len(pre_idx) == 0 or len(raw_chunk_idx) == 0:
        return
    passo = pd.Series(pre_idx).diff().median() if len(pre_idx) > 1 else pd.NaT
    if pd.isna(passo):
        passo = pd.Series(raw_chunk_idx).diff().median()
    tol = (passo / 2 if pd.notna(passo) else pd.Timedelta(0)) + pd.Timedelta(milliseconds=1)
    sobrev = pd.Series(True, index=pre_idx[~pre_idx.duplicated()])
    ok = sobrev.reindex(raw_chunk_idx, method="nearest", tolerance=tol).notna().to_numpy()
    mascara.loc[raw_chunk_idx[ok]] = True


class Persistencia:
    def __init__(self, n: int = 2):
        self.n, self.ultimo, self.cont = n, None, 0

    def update(self, contexto: str) -> str:
        if contexto in CONTEXTOS_FALHA:
            self.cont = self.cont + 1 if contexto == self.ultimo else 1
        else:
            self.cont = 0
        self.ultimo = contexto
        if contexto in CONTEXTOS_FALHA and self.cont < self.n:
            return "suspeita_" + contexto        # ainda não confirmou
        return contexto


def limiar_robusto(ref_errors, n_mad: float = 6.0, percentil_piso: float = 99.5) -> float:
    r = np.asarray(ref_errors, dtype=float)
    r = r[np.isfinite(r) & (r >= 0)]
    eps = 1e-3 * (float(np.median(r)) or 1.0)
    l = np.log(r + eps)
    mu = float(np.median(l))
    sd = 1.4826 * float(np.median(np.abs(l - mu))) or float(l.std()) or 1e-6
    thr = float(np.exp(mu + n_mad * sd) - eps)
    return max(thr, float(np.percentile(r, percentil_piso)))

def drift_triggered_walkforward(
    df_pre: pd.DataFrame,
    equipment_id: str,
    model_type: str,
    threshold_percentile: float,
    detector_name: str,
    device: str,
    baseline: Era,
    p: Params,
    preset: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps = get_preprocessing_steps(equipment_id, preset=preset)
    inicio, fim = df_pre.index.min(), df_pre.index.max()
    spd = samples_per_day_de(df_pre)
    step_s = 86400.0 / spd
    jan = janelas_prefalha(p.failure_events, p.prefailure_days)

    if not sensores_degradacao(df_pre.columns, p.deg_patterns):
        print(f"  [{detector_name}] [aviso] nenhuma coluna casa com {list(p.deg_patterns)}: "
              f"sem sensor físico, todo drift será 'drift_provavel' (nunca bloqueia por sensor).")
    ausentes = [c for c in p.regime_cols if c not in df_pre.columns]
    if ausentes:
        print(f"  [{detector_name}] [aviso] coluna(s) de regime ausente(s): {ausentes}. "
              f"O gate fica menos preciso.")

    era_id = 0
    era = baseline
    detector = build_detector(detector_name, era.ref_errors, spd)
    min_era_days = p.min_era_days
    if min_era_days is None:
        min_era_days = max(auto_min_era_days(detector_name, detector, df_pre), DEFAULT_MIN_ERA_DAYS)
        print(f"  [{detector_name}] idade mínima de era = {min_era_days} dia(s)")

    corte_inicial = inicio + pd.Timedelta(days=p.initial_train_days)
    retrain_log: list[dict] = [{
        "detector": detector_name, "era": 0, "era_start": inicio, "retrain_at": inicio,
        "motivo": "baseline_inicial", "n_amostras_treino": int((df_pre.index < corte_inicial).sum()),
        "threshold": era.threshold,
    }]
    drift_events: list[dict] = []
    chunk_rows: list[dict] = []
    parts: list[pd.DataFrame] = []

    cursor, era_start = corte_inicial, corte_inicial
    suspeita: dict | None = None        
    persist_ctx = Persistencia(p.persist_contexto_n)
    mascara_op: pd.Series | None = None
    if df_pre.index.is_unique:
        mascara_op = pd.Series(df_pre.index < corte_inicial, index=df_pre.index)
    sem_dados_ini: pd.Timestamp | None = None
    sem_dados_n = 0
    ultimo_adiamento = None               
    pausa_ate: pd.Timestamp | None = None 
    pausa_motivo = ""
    bloqueio_inicio: pd.Timestamp | None = None
    bloqueio_ultimo: pd.Timestamp | None = None
    lookback = pd.Timedelta(hours=p.lookback_horas)

    while cursor < fim:
        prox = min(cursor + pd.Timedelta(days=p.chunk_days), fim)
        ultimo = prox >= fim
        raw = _fatia(df_pre, cursor - lookback, prox, incluir_fim=ultimo)
        n_raw = int((raw.index >= cursor).sum())
        motivo_vazio = None
        pedaco_df = None
        if n_raw == 0:
            motivo_vazio = "sem_linhas_brutas"
        else:
            try:
                pedaco_df, _, _ = run_preprocessing(
                    raw, steps, fitted_artifacts=era.artifacts,
                    return_artifacts=True, return_report=True)
                pedaco_df = pedaco_df[pedaco_df.index >= cursor]
                if len(pedaco_df) == 0:
                    motivo_vazio = "preprocessing_removeu_tudo"
            except ValueError as exc:
                if "0 sample" in str(exc) or "empty" in str(exc).lower():
                    motivo_vazio = "preprocessing_removeu_tudo"
                else:
                    print(f"  [{detector_name}] [aviso] chunk {cursor.date()} → {prox.date()} falhou no "
                          f"preprocessing (ValueError: {exc}) — pulando.")
                    motivo_vazio = "erro_preprocessing"
            except Exception as exc:
                print(f"  [{detector_name}] [aviso] chunk {cursor.date()} → {prox.date()} falhou no "
                      f"preprocessing ({type(exc).__name__}: {exc}) — pulando.")
                motivo_vazio = "erro_preprocessing"

        if motivo_vazio is not None:
            if sem_dados_ini is None:
                sem_dados_ini, sem_dados_n = cursor, 0
                chunk_diag = raw[raw.index >= cursor]
                nan_pct = (100 * chunk_diag.isna().mean()).round(0)
                reg_med = {c: round(float(chunk_diag[c].median()), 3)
                           for c in p.regime_cols if c in chunk_diag.columns and chunk_diag[c].notna().any()}
                print(f"  [{detector_name}] {cursor.date()}: chunk sem dados úteis ({motivo_vazio}; "
                      f"{n_raw} linhas brutas; regime(mediana)={reg_med or 'n/d'}; "
                      f"colunas 100% NaN={sorted(nan_pct[nan_pct >= 100].index.tolist()) or 'nenhuma'}) — "
                      f"pulado. Se for equipamento parado, confira o filtro de operação do preset.")
            sem_dados_n += 1
            chunk_rows.append({
                "inicio": cursor, "data": prox, "era": era_id, "n_pontos": 0, "n_linhas_brutas": n_raw,
                "disparo_detector": False, "suspeita_ativa": suspeita is not None,
                "decisao": "", "status": "sem_dados", "contexto": "sem_dados",
                "motivo_sem_dados": motivo_vazio})
            cursor = prox
            continue
        if sem_dados_ini is not None:
            print(f"  [{detector_name}] {cursor.date()}: dados voltaram após {sem_dados_n} chunk(s) "
                  f"sem dados úteis (desde {sem_dados_ini.date()}).")
            sem_dados_ini = None

        if mascara_op is not None:
            marcar_operacao(mascara_op, raw.index[raw.index >= cursor], pedaco_df.index)

        erros = erros_modelo(era.model, model_type, pedaco_df, threshold_percentile, device)
        med_ref = max(float(np.median(era.ref_errors)), 1e-12)
        parts.append(pd.DataFrame({
            "reconstruction_error": erros.to_numpy(), "exceeds": (erros > era.threshold).to_numpy(),
            "era": era_id,
            "erro_rel": erros.to_numpy() / med_ref,
            "contexto": "normal"}, index=erros.index))

        ctx = {"contexto": "normal", "exc_pct_janela": 0.0, "regimes_mudaram": "",
               "sensores_fora": "", "forma": "-"}
        ctx_conf = "normal"
        if p.gate_continuo:
            try:
                n_back = int(p.gate_janela_dias / max(p.chunk_days, 1)) + 2
                exc_hist = pd.concat([q["exceeds"] for q in parts[-n_back:]])
                ctx = classificar_contexto(avaliar_regime_novo, df_pre, era, exc_hist, prox, p,
                                           mascara_operacao=mascara_op)
            except Exception as exc:
                print(f"  [{detector_name}] [aviso] gate contínuo falhou em {prox.date()} "
                      f"({type(exc).__name__}: {exc}); contexto = indeterminado.")
                ctx["contexto"] = "indeterminado"
            ctx_conf = persist_ctx.update(ctx["contexto"])
            parts[-1]["contexto"] = ctx["contexto"]

        em_pausa = pausa_ate is not None and prox <= pausa_ate
        t_disparo = None
        if em_pausa:
            detector.reset()
        else:
            if pausa_ate is not None:
                pausa_ate, pausa_motivo = None, ""
            for ts, v in erros.items():
                if detector.update(float(v)) and t_disparo is None:
                    t_disparo = ts

            if t_disparo is not None:
                if suspeita is None:
                    suspeita = {"inicio": t_disparo, "ultimo": t_disparo, "n": 1}
                else:
                    suspeita["ultimo"], suspeita["n"] = t_disparo, suspeita["n"] + 1
                detector.rearm()
            elif suspeita is not None and len(erros) and \
                    not voltou_ao_normal(erros, era.ref_errors, p.fator_nivel_normal):
                suspeita["ultimo"], suspeita["n"] = erros.index[-1], suspeita["n"] + 1
            elif suspeita is not None and \
                    (prox - suspeita["ultimo"]) >= pd.Timedelta(days=p.expira_suspeita_dias):
                suspeita = None    # não persistiu: era ruído
                detector.reset()

        idade_era_dias = (prox - era_start).days
        confirmado = (
            suspeita is not None and suspeita["n"] >= p.confirm_disparos
            and suspeita["ultimo"] >= cursor
            and (suspeita["ultimo"] - suspeita["inicio"]) >= pd.Timedelta(days=p.confirm_dias)
            and idade_era_dias >= min_era_days
        )

        decisao = ""
        nivel_normal = False
        if confirmado:
            rec = pd.concat(parts).sort_index()
            rec = rec[(rec["era"] == era_id) & (rec.index > prox - pd.Timedelta(days=max(p.confirm_dias, 1.0)))]
            nivel_normal = len(rec) > 0 and voltou_ao_normal(
                rec["reconstruction_error"], era.ref_errors, p.fator_nivel_normal)
        if confirmado and nivel_normal:
            drift_events.append({
                "detector": detector_name, "data": prox, "era": era_id,
                "idade_era_dias": idade_era_dias, "primeiro_disparo": suspeita["inicio"],
                "n_disparos": suspeita["n"], "decisao": "alarme_falso_descartado"})
            decisao = "alarme_falso_descartado"
            suspeita = None
            detector.reset()
        elif confirmado:
            hist = pd.concat(parts).sort_index()
            hist["is_anomaly"] = kofn(hist["exceeds"], p.persist_k, p.persist_n)
            diag = diagnosticar_drift(df_pre, hist, prox, era.artifacts, p.deg_patterns,
                                      p.regime_cols, p.deg_window_days, p.deg_slope_rel)

            erros_era = hist.loc[hist["era"] == era_id, "reconstruction_error"]
            t_mud = estimar_inicio_regime(erros_era, era.ref_errors)
            dias_regime_novo = (prox - t_mud).days
            train_start = max(inicio, t_mud + pd.Timedelta(days=p.buffer_transicao_dias))

            av = None
            if era.train_ini is not None and (prox - train_start).days >= p.min_dias_diagnostico:
                av = avaliar_regime_novo(df_pre, era.train_ini, era.train_fim, train_start, prox,
                                         p.regime_cols, p.deg_patterns, p.regime_delta_min,
                                         p.resid_min, p.tend_min, p.extrap_resid_fator)

            rec_g = hist[hist.index >= prox - pd.Timedelta(days=p.gate_janela_dias)]
            exc_pct = 100.0 * float(rec_g["exceeds"].mean()) if len(rec_g) else 0.0
            anomalia_ativa = bool(rec_g["is_anomaly"].any()) or exc_pct >= p.gate_exceeds_pct
            diag_pendente = av is None and (prox - train_start).days < p.min_dias_diagnostico

            if av is None:
                if diag_pendente:
                    motivo_bloq = ""
                else: 
                    motivo_bloq = "tendencia_30d" if diag["conclusao"] == "possivel_degradacao" else ""
            elif av["sensores_fora"]:
                motivo_bloq = "sensor_fora_do_regime" 
            elif p.gate_anomalia and anomalia_ativa and not av["regime_mudou"]:
                motivo_bloq = "anomalia_ativa_sem_regime" 
            else:
                motivo_bloq = ""
            bloquear = p.bloquear_degradacao and bool(motivo_bloq)

            evento = {
                "detector": detector_name, "data": prox, "era": era_id,
                "idade_era_dias": idade_era_dias,
                "primeiro_disparo": suspeita["inicio"], "n_disparos": suspeita["n"],
                "dias_ate_confirmar": (suspeita["ultimo"] - suspeita["inicio"]).total_seconds() / 86400.0,
                **{k: v for k, v in diag.items() if k != "tendencias"},
                "mudanca_estimada": t_mud, "dias_regime_novo": dias_regime_novo,
                "anomalia_ativa": anomalia_ativa, "exceeds_pct_recente": exc_pct,
                "motivo_bloqueio": motivo_bloq, "diagnostico_pendente": diag_pendente,
                **({"delta_regime_sd": av["delta_regime_sd"], "regime_mudou": av["regime_mudou"],
                    "regimes_mudaram": ", ".join(av["regimes_mudaram"]),
                    "sensores_fora": ", ".join(av["sensores_fora"]),
                    "resid_max_sd": av["resid_max_sd"], "tend_max_sd": av["tend_max_sd"],
                    "forma_desvio": av["forma"], "extrapolacao": av["extrapolacao"],
                    "resid_lim_usado": av["resid_lim_usado"]} if av else {}),
                "decisao": "retreinou",
            }
            drift_events.append(evento)

            if bloquear:
                if bloqueio_ultimo is not None and \
                        (prox - bloqueio_ultimo).days > 2 * p.cooldown_bloqueio_dias:
                    bloqueio_inicio = None 
                bloqueio_inicio = bloqueio_inicio or prox
                bloqueio_ultimo = prox
                evento["dias_em_bloqueio"] = (prox - bloqueio_inicio).days
                escalado = evento["dias_em_bloqueio"] >= p.teto_bloqueio_dias
                evento["decisao"] = decisao = ("bloqueio_escalado_operacao" if escalado
                                               else "retreino_bloqueado_degradacao")
                print(f"  [{detector_name}] {prox.date()}: {motivo_bloq} "
                      f"(forma={evento.get('forma_desvio', '-')}) → "
                      f"{'ESCALADO À OPERAÇÃO (teto de bloqueio)' if escalado else 'adaptação bloqueada'}; "
                      f"{evento['dias_em_bloqueio']} dia(s) em bloqueio.")
                detector.reset()
                suspeita = None
                pausa_ate = prox + pd.Timedelta(days=p.cooldown_bloqueio_dias)
                pausa_motivo = "degradacao"
            else:
                bloqueio_inicio = None
                if dias_regime_novo < p.min_dias_regime_novo or diag_pendente:
                    evento["decisao"] = decisao = "retreino_adiado_regime_curto"
                    chave = suspeita["inicio"] 
                    if ultimo_adiamento != chave or dias_regime_novo % 7 == 0:
                        print(f"  [{detector_name}] {prox.date()}: mudança em {t_mud.date()}, só "
                              f"{dias_regime_novo}d de regime novo (< {p.min_dias_regime_novo}"
                              f"{', diagnóstico pendente' if diag_pendente else ''}) → ADIADO.")
                    ultimo_adiamento = chave
                    detector.rearm() 
                    _mesclar_adiamento(drift_events)
                else:
                    erros_brutos = erros_era[erros_era.index >= train_start]
                    erros_novos, frac_rem = erros_brutos, 0.0
                    if p.gate_anomalia:   
                        erros_novos, frac_rem = filtrar_anomalias_erros(
                            erros_brutos, hist, p.margem_anomalia_horas)
                    dec = decidir_recalibracao(erros_novos, era.ref_errors, p.alarm_sigma,
                                               threshold_percentile, spd, p.limite_leve,
                                               robusto=p.limiar_robusto, n_mad=p.limiar_mad)
                    if dec["acao"] == "recalibrar" and frac_rem > p.max_frac_filtrada_recal:
                        dec = {**dec, "acao": "retreinar",
                               "motivo": f"{frac_rem:.0%} dos erros novos eram anomalia"}
                    evento["erro_rel_regime_novo"] = dec["erro_rel"]
                    evento["frac_anomalia_removida_recal"] = frac_rem

                    if p.permitir_recalibracao and dec["acao"] == "recalibrar":
                        cont = metricas_contaminacao(
                            erros_brutos.index, hist,
                            thr_rel(dec["threshold"], dec["ref_errors"]),
                            thr_rel(era.threshold, era.ref_errors),
                            thr_rel(baseline.threshold, baseline.ref_errors), jan)
                        evento.update(cont)
                        era_id += 1
                        era = replace(era, ref_errors=dec["ref_errors"], threshold=dec["threshold"],
                                      train_ini=train_start, train_fim=prox)
                        detector = build_detector(detector_name, era.ref_errors, spd)
                        retrain_log.append({
                            "detector": detector_name, "era": era_id, "era_start": prox,
                            "retrain_at": prox, "mudanca_estimada": t_mud,
                            "dias_regime_novo": dias_regime_novo,
                            "motivo": f"recalibracao_{detector_name}", "n_amostras_treino": 0,
                            "erro_rel_regime_novo": dec["erro_rel"], "threshold": era.threshold,
                            "frac_anomalia_removida_recal": frac_rem, **cont,
                        })
                        era_start, suspeita = prox, None
                        evento["decisao"] = decisao = "recalibrou_regua"
                        print(f"  [{detector_name}] {prox.date()}: drift leve "
                              f"({dec['erro_rel']:.2f}x) → régua RECALIBRADA, modelo mantido.")
                    else:
                        fim_treino = prox - pd.Timedelta(days=p.quarentena_dias)
                        fatia = _fatia(df_pre, train_start, fim_treino)
                        excluir = (intervalos_episodios_curtos(
                                       hist, p.max_dias_episodio_excluir, p.margem_anomalia_horas)
                                   if p.excluir_anomalias_treino else [])
                        try:
                            nova = treinar_era(
                                fatia, steps, model_type, device, threshold_percentile,
                                p.alarm_sigma, spd, p.seed + era_id + 1, p.model_kwargs,
                                f"{detector_name}|retreino_em_{prox.date()}", excluir=excluir,
                                robusto=p.limiar_robusto, n_mad=p.limiar_mad)
                        except ValueError as exc:
                            evento["decisao"] = decisao = "retreino_adiado_dados_insuficientes"
                            print(f"  [{detector_name}] {prox.date()}: retreino adiado ({exc})")
                            detector.rearm()
                            _mesclar_adiamento(drift_events)
                            nova = None

                        if nova is not None:
                            evento["amostras_anomalas_removidas"] = nova.n_excluidas
                            cont = metricas_contaminacao(
                                nova.train_index, hist,
                                thr_rel(nova.threshold, nova.ref_errors),
                                thr_rel(era.threshold, era.ref_errors),
                                thr_rel(baseline.threshold, baseline.ref_errors), jan)
                            evento.update(cont)
                            ganho, fonte = np.nan, "indisponivel"
                            try:
                                ganho = ganho_relativo(
                                    df_pre, steps, era, nova, model_type, threshold_percentile,
                                    device, fim_treino, prox, p.lookback_horas, spd)
                                fonte = "holdout_quarentena"
                                if ganho != ganho: 
                                    ganho = ganho_relativo(
                                        df_pre, steps, era, nova, model_type, threshold_percentile,
                                        device, nova.val_index.min(),
                                        nova.val_index.max() + pd.Timedelta(seconds=1),
                                        p.lookback_horas, 20)
                                    fonte = "validacao_interna"
                            except Exception as exc:
                                print(f"    [aviso] não deu para comparar com o modelo vigente "
                                      f"({type(exc).__name__}: {exc}); aceitando o candidato.")
                            evento["ganho_candidato"] = ganho
                            evento["ganho_fonte"] = fonte

                            if ganho == ganho and ganho < p.min_ganho_retreino:
                                evento["decisao"] = decisao = "retreino_rejeitado_sem_ganho"
                                print(f"  [{detector_name}] {prox.date()}: candidato só {ganho:.2f}x melhor "
                                      f"(< {p.min_ganho_retreino}x, {fonte}) → REJEITADO, modelo mantido.")
                                detector.reset()
                                suspeita = None
                                pausa_ate = prox + pd.Timedelta(days=p.cooldown_rejeicao_dias)
                                pausa_motivo = "rejeicao"
                            else:
                                era_id += 1
                                era = nova
                                detector = build_detector(detector_name, era.ref_errors, spd)
                                n_treino = len(nova.train_index)
                                horas = n_treino * step_s / 3600.0
                                retrain_log.append({
                                    "detector": detector_name, "era": era_id, "era_start": prox,
                                    "train_start": train_start, "retrain_at": prox,
                                    "mudanca_estimada": t_mud, "dias_regime_novo": dias_regime_novo,
                                    "motivo": f"drift_{detector_name}",
                                    "n_amostras_treino": n_treino,
                                    "horas_operacao_treino": round(horas, 1),
                                    "atende_politica_horas": bool(horas >= p.min_horas_politica),
                                    "ganho_candidato": ganho, "ganho_fonte": fonte,
                                    "threshold": era.threshold, **cont,
                                })
                                era_start, suspeita = prox, None
                                evento["decisao"] = decisao = "retreinou"
                                tfa = cont["treino_frac_anomalia"]
                                print(f"  [{detector_name}] {prox.date()}: mudança de conceito → "
                                      f"RETREINADO ({n_treino} amostras, ganho {ganho:.2f}x [{fonte}], "
                                      f"{100 * tfa:.1f}% anômalas no treino)." if tfa == tfa else
                                      f"  [{detector_name}] {prox.date()}: mudança de conceito → "
                                      f"RETREINADO ({n_treino} amostras, ganho {ganho:.2f}x [{fonte}]).")

        ex = pd.concat([q["exceeds"] for q in parts[-2:]])
        anom_chunk = kofn(ex, p.persist_k, p.persist_n).iloc[-len(erros):]
        anom_pct = 100.0 * float(anom_chunk.mean())

        if anom_chunk.any() or decisao in (
                "retreino_bloqueado_degradacao", "bloqueio_escalado_operacao", "retreinou") \
                or pausa_motivo == "degradacao":
            status = "vermelho"
        elif decisao or suspeita is not None or pausa_motivo \
                or ctx["contexto"] in ("regime_novo", "falha_sensor", "anomalia_sem_regime"):
            status = "amarelo"
        else:
            status = "verde"
        chunk_rows.append({
            "inicio": cursor, "data": prox, "era": era_id, "n_pontos": len(erros),
            "excede_limiar_pct": 100.0 * float((erros > era.threshold).mean()),
            "erro_p50_rel": float(erros.median()) / max(float(np.median(era.ref_errors)), 1e-12),
            "disparo_detector": t_disparo is not None, "anomalia_chunk_pct": anom_pct,
            "suspeita_ativa": suspeita is not None, "decisao": decisao, "status": status,
            "contexto": ctx["contexto"], "contexto_confirmado": ctx_conf,
            "exc_pct_janela": ctx["exc_pct_janela"], "regimes_mudaram_ctx": ctx["regimes_mudaram"],
            "sensores_fora_ctx": ctx["sensores_fora"], "forma_ctx": ctx["forma"],
        })
        cursor = prox

    cols = ["reconstruction_error", "exceeds", "is_anomaly", "era"]
    if parts:
        full_scores = pd.concat(parts).sort_index()
        full_scores["is_anomaly"] = kofn(full_scores["exceeds"], p.persist_k, p.persist_n)
    else:
        full_scores = pd.DataFrame(columns=cols + ["erro_rel", "contexto"])

    chunk_log = pd.DataFrame(chunk_rows)
    if len(chunk_log) and len(full_scores):
        idx = full_scores.index
        an = full_scores["is_anomaly"].to_numpy(dtype=float)
        pct = []
        for a, b in zip(chunk_log["inicio"], chunk_log["data"]):
            i0, i1 = idx.searchsorted(a), idx.searchsorted(b)
            pct.append(100.0 * float(an[i0:i1].mean()) if i1 > i0 else np.nan)
        chunk_log["alarme_anomalia_pct"] = pct
    return full_scores, pd.DataFrame(retrain_log), pd.DataFrame(drift_events), chunk_log

def pontuar_resultado(full_scores, config, prefailure_days: int, normal_end_days: int,
                      coluna: str = "is_anomaly") -> dict[str, Any]:
    failure_events = getattr(config, "failure_events", None)
    failure_date = getattr(config, "failure_date", None)
    if coluna not in full_scores:
        coluna = "is_anomaly"
    scores_df = full_scores[["reconstruction_error", coluna]].rename(columns={coluna: "is_anomaly"})

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


def rodar_sem_adaptacao(df_pre, equipment_id, model_type, preset, device,
                        threshold_percentile, baseline: Era, p: Params, config,
                        prefailure_days: int, normal_end_days: int):
    steps = get_preprocessing_steps(equipment_id, preset=preset)
    corte = df_pre.index.min() + pd.Timedelta(days=p.initial_train_days)
    resto, _, _ = run_preprocessing(
        df_pre[df_pre.index >= corte], steps, fitted_artifacts=baseline.artifacts,
        return_artifacts=True, return_report=True)
    serie = erros_modelo(baseline.model, model_type, resto, threshold_percentile, device)
    fs = pd.DataFrame({"reconstruction_error": serie.to_numpy(),
                       "exceeds": (serie > baseline.threshold).to_numpy(),
                       "era": 0}, index=serie.index)
    fs["is_anomaly"] = kofn(fs["exceeds"], p.persist_k, p.persist_n)
    return fs, pontuar_resultado(fs, config, prefailure_days, normal_end_days), corte

def plotar_timeline(full_scores, retrain_log, output_path, failure_events=None,
                    title: str = "", ax=None, eventos: pd.DataFrame | None = None,
                    mostrar_threshold: bool = True) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    if not full_scores.empty:
        ax.plot(full_scores.index, full_scores["reconstruction_error"],
                color="steelblue", linewidth=0.7, alpha=0.8, label="Erro de reconstrução")
        anomalias = full_scores[full_scores["is_anomaly"].astype(bool)]
        if not anomalias.empty:
            ax.scatter(anomalias.index, anomalias["reconstruction_error"], color="black", s=14,
                       zorder=4, edgecolor="none", label=f"Anomalia ({len(anomalias)})")
        if mostrar_threshold:
            for _, row in retrain_log.iterrows():
                era_pontos = full_scores[full_scores["era"] == row["era"]]
                if era_pontos.empty:
                    continue
                ax.hlines(row["threshold"], row["retrain_at"], era_pontos.index.max(), color="gray",
                          linestyle=":", linewidth=1, label="Threshold" if row["era"] == 0 else None)

    motivos = retrain_log["motivo"].astype(str)
    mudancas = retrain_log[motivos.str.startswith(("drift_", "recalibracao_"))]
    for i, (_, row) in enumerate(mudancas.iterrows()):
        ax.axvline(row["retrain_at"], color="red", linestyle="--", linewidth=1.3,
                   label="Mudança de conceito" if i == 0 else None)

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


def plotar_comparativo(resultados: dict[str, tuple], output_path: Path,
                       failure_events=None, title: str = "") -> None:
    n = len(resultados)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(14, 2.6 * n))
    if n == 1:
        axes = [axes]
    for ax, (nome, (fs, lg, ev)) in zip(axes, resultados.items()):
        n_acoes = max(len(lg) - 1, 0)
        plotar_timeline(fs, lg, None, failure_events=failure_events,
                        title=f"{nome} — {n_acoes} retreino(s)/recalibração(ões)", ax=ax, eventos=ev)
    axes[-1].set_xlabel("Tempo")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

def _conta(df: pd.DataFrame, coluna: str, valor: str) -> int:
    return int((df[coluna] == valor).sum()) if len(df) and coluna in df else 0


def _max_col(df: pd.DataFrame, coluna: str) -> float:
    if coluna in df and df[coluna].notna().any():
        return float(df[coluna].max())
    return float("nan")


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipment", required=True, choices=list(EQUIPMENT_CONFIGS.keys()))
    ap.add_argument("--local-data", action="store_true")
    ap.add_argument("--models", nargs="+", default=None, choices=["dense", "lstm", "ocsvm", "iforest"],
                    help="Quais modelos testar (default: os 4)")
    ap.add_argument("--presets", nargs="+", default=None,
                    help="Presets de preprocessing (default: todos do equipamento)")
    ap.add_argument("--detectors", nargs="+", default=None, choices=DETECTOR_NAMES,
                    help="Técnicas de drift (default: todas)")
    ap.add_argument("--threshold-percentile", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=0, help="Semente de treino (reprodutibilidade).")

    g = ap.add_argument_group("ciclo de produção")
    g.add_argument("--initial-train-days", type=int, default=None,
                   help="Dias do baseline inicial. Default: curva de aprendizado por preset.")
    g.add_argument("--min-era-days", type=int, default=None,
                   help=f"Idade mínima da era antes de aceitar drift. Default: máx(aquecimento, {DEFAULT_MIN_ERA_DAYS}).")
    g.add_argument("--chunk-days", type=int, default=1,
                   help="Tamanho do lote de dados novos (default 1 dia, como em produção).")
    g.add_argument("--lookback-horas", type=float, default=6.0,
                   help="Contexto antes do chunk para os passos temporais do preprocessing.")
    g.add_argument("--alarm-sigma", type=float, default=DEFAULT_ALARM_SIGMA,
                   help="Limiar de anomalia = média + N*desvio da referência. 0 = usa --threshold-percentile.")
    g.add_argument("--persist-k", type=int, default=DEFAULT_PERSIST_K)
    g.add_argument("--persist-n", type=int, default=DEFAULT_PERSIST_N)

    g = ap.add_argument_group("confirmação do drift (por tempo, independe do chunk)")
    g.add_argument("--confirm-dias", type=float, default=3.0,
                   help="Dias entre o 1º e o último disparo para confirmar.")
    g.add_argument("--confirm-disparos", type=int, default=2)
    g.add_argument("--expira-suspeita-dias", type=float, default=10.0,
                   help="Sem novo disparo por tantos dias, a suspeita é descartada.")

    g = ap.add_argument_group("falha física × mudança de conceito")
    g.add_argument("--deg-patterns", nargs="+", default=list(DEG_PADROES_DEFAULT))
    g.add_argument("--regime-cols", nargs="+", default=list(REGIME_COLS_DEFAULT),
                   help="Variáveis operacionais que definem o regime "
                        "(ex.: Corrente Vazao Temp_ambiente). Nunca são tratadas como sensor de desgaste. "
                        "Nomes com espaço: use aspas (--regime-cols \"B-x: Pressão Descarga\" "
                        "\"B-x: Pressão Sucção\") ou ';' entre eles; sem aspas o script tenta "
                        "reconstruir os nomes a partir das colunas do dado.")
    g.add_argument("--deg-window-days", type=int, default=30)
    g.add_argument("--deg-slope-rel", type=float, default=1.0)
    g.add_argument("--permitir-retreino-em-degradacao", action="store_true",
                   help="Só registra o diagnóstico; NÃO bloqueia a adaptação (para comparar).")
    g.add_argument("--cooldown-bloqueio-dias", type=int, default=14,
                   help="Após um bloqueio, silencia o detector por esse tempo "
                        "(evita um evento novo a cada chunk).")
    g.add_argument("--sem-gate-anomalia", action="store_true",
                   help="Desliga o gate de anomalia ativa e o filtro de anomalias na recalibração (comparação).")
    g.add_argument("--gate-janela-dias", type=int, default=7,
                   help="Janela recente (dias) para decidir se há anomalia ativa.")
    g.add_argument("--gate-exceeds-pct", type=float, default=3.0,
                   help="%% de pontos acima do limiar na janela que conta como anomalia ativa (pega falha intermitente).")
    g.add_argument("--teto-bloqueio-dias", type=int, default=45,
                   help="Dias seguidos em bloqueio até escalar à operação (sem adaptação automática).")
    g.add_argument("--min-dias-diagnostico", type=int, default=7,
                   help="Dias mínimos de regime novo para o diagnóstico sensor~regime. "
                        "Com menos que isso o drift é ADIADO (não bloqueia pela tendência de 30 dias).")
    g.add_argument("--regime-delta-min", type=float, default=1.0,
                   help="|Δ variável de regime| (desvios robustos da referência) que conta como mudança de regime.")
    g.add_argument("--resid-min", type=float, default=2.0,
                   help="Resíduo do sensor (dado o regime), em desvios, que conta como sensor fora do previsto.")
    g.add_argument("--tend-min", type=float, default=2.0,
                   help="Subida do resíduo ao longo da janela, em desvios, que conta como rampa.")
    g.add_argument("--sem-gate-continuo", action="store_true",
                   help="Desliga a classificação contínua (regime_novo / falha_sensor) a cada chunk.")
    g.add_argument("--persist-contexto-n", type=int, default=2,
                   help="Chunks seguidos com o mesmo contexto de falha para confirmar a falha.")
    g.add_argument("--limiar-robusto", action="store_true",
                   help="Limiar = mediana + N*MAD no log do erro, em vez de média + N*desvio.")
    g.add_argument("--limiar-mad", type=float, default=6.0, help="N do --limiar-robusto.")
    g.add_argument("--extrap-resid-fator", type=float, default=2.0,
                   help="Multiplica --resid-min quando o regime novo está fora da faixa do treino "
                        "(o ajuste linear sensor~regime extrapola e é menos confiável).")

    g = ap.add_argument_group("ação sobre mudança de conceito")
    g.add_argument("--min-dias-regime-novo", type=int, default=DEFAULT_MIN_DIAS_REGIME_NOVO)
    g.add_argument("--buffer-transicao-dias", type=int, default=DEFAULT_BUFFER_TRANSICAO_DIAS)
    g.add_argument("--limite-leve", type=float, default=DEFAULT_LIMITE_LEVE,
                   help="Erro relativo (mediana do regime novo / mediana da referência) até o qual "
                        "só se recalibra a régua.")
    g.add_argument("--fator-nivel-normal", type=float, default=DEFAULT_FATOR_NIVEL_NORMAL,
                   help="Mediana recente <= N x mediana da referência = o erro voltou ao normal "
                        "(descarta alarme falso). Deve ser menor que --limite-leve.")
    g.add_argument("--sem-recalibracao", action="store_true",
                   help="Desliga a recalibração: todo drift de regime vira retreino (comparação).")
    g.add_argument("--min-ganho-retreino", type=float, default=1.2,
                   help="O modelo novo precisa reduzir o erro do regime novo em N vezes (mediana, "
                        "em escala relativa à referência de cada modelo) vs. o vigente; senão é "
                        "rejeitado. 0 desliga a checagem.")
    g.add_argument("--cooldown-rejeicao-dias", type=int, default=7)
    g.add_argument("--quarentena-dias", type=int, default=2,
                   help="O retreino não usa os últimos N dias antes da confirmação do drift "
                        "(essa janela serve de holdout para medir o ganho do candidato).")
    g.add_argument("--manter-anomalias-no-treino", action="store_true",
                   help="Não remove episódios curtos de anomalia do treino do retreino.")
    g.add_argument("--max-dias-episodio-excluir", type=float, default=3.0)
    g.add_argument("--margem-anomalia-horas", type=float, default=24.0)
    g.add_argument("--min-horas-politica", type=float, default=4000.0,
                   help="Horas de operação da política de retreino (só registra se o treino atenderia).")

    g = ap.add_argument_group("benchmark do detector (opcional)")
    g.add_argument("--controle-fim", default=None,
                   help="Fim de um período SEM drift após o baseline (AAAA-MM-DD).")
    g.add_argument("--drift-time", action="append", default=None,
                   help="Início real de um drift (AAAA-MM-DD). Repetível.")
    g.add_argument("--horizonte-dias", type=int, default=90)

    ap.add_argument("--prefailure-days", type=int, default=30)
    ap.add_argument("--normal-end-days", type=int, default=60)
    ap.add_argument("--output-dir", default="drift_retrain_out")
    ap.add_argument("--max-fp-rate", type=float, default=0.0,
                    help="Taxa máxima de FP (0-1); acima disso a combinação é marcada como reprovada. 0 desliga.")

    ap.add_argument("--remote", action="store_true", help="Envia para execução remota no ClearML e sai")
    ap.add_argument("--queue", default="default")
    ap.add_argument("--clearml-project", default="Transpetro")
    ap.add_argument("--no-clearml-upload", action="store_true")
    ap.add_argument("--no-clearml", action="store_true", help="Não usa ClearML (debug local)")
    args = ap.parse_args()

    if args.limite_leve <= args.fator_nivel_normal:
        print(f"[aviso] --limite-leve ({args.limite_leve}) <= --fator-nivel-normal "
              f"({args.fator_nivel_normal}): a recalibração nunca vai disparar.")

    task = None
    if not args.no_clearml:
        from clearml import Task
        Task.add_requirements("setuptools>=65.0")
        Task.add_requirements("gitpython>=3.1.40")
        Task.add_requirements("pyarrow")
        Task.add_requirements("torch", package_version="")
        task = Task.init(project_name=args.clearml_project,
                         task_name=f"drift_retrain_{args.equipment}_grid",
                         output_uri=True, reuse_last_task_id=False)
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
    spd = samples_per_day_de(df_pre)

    args.regime_cols = reconstruir_nomes(args.regime_cols, df_pre.columns, exato=True)
    args.deg_patterns = reconstruir_nomes(args.deg_patterns, df_pre.columns, exato=False)
    print(f"  Variáveis de regime: {args.regime_cols}\n  Padrões de desgaste: {args.deg_patterns}")

    deg_cols = sensores_degradacao(df_pre.columns, args.deg_patterns)
    print(f"  Colunas tratadas como desgaste: {deg_cols}")
    n_temp = [c for c in deg_cols if "temperatura" in str(c).lower()]
    if len(n_temp) != 1:
        print(f"  [aviso] esperado 1 temperatura de degradação, casou {len(n_temp)}: {n_temp}")
        
    for c in args.regime_cols:
        if c not in df_pre.columns:
            sug = difflib.get_close_matches(c, [str(x) for x in df_pre.columns], n=3, cutoff=0.5)
            print(f"  [aviso] --regime-cols '{c}' não existe no dado. Parecidas: {sug or 'nenhuma'}. "
                  f"Use aspas por nome: --regime-cols \"nome 1\" \"nome 2\"")

    model_names = args.models or ["dense", "lstm", "ocsvm", "iforest"]
    available_presets = (list(config.preprocess_presets.keys())
                         if getattr(config, "preprocess_presets", None) else ["baseline"])
    preset_names = args.presets or available_presets
    detector_names = args.detectors or list(DETECTOR_NAMES)

    total = len(model_names) * len(preset_names) * len(detector_names)
    print(f"\nGrade: {len(model_names)} modelo(s) x {len(preset_names)} preset(s) x "
          f"{len(detector_names)} técnica(s) = {total} combinações (+ controle sem adaptação por modelo/preset)")
    print(f"  Modelos: {model_names}\n  Presets: {preset_names}\n  Técnicas: {detector_names}")
    if total > 20:
        print("  [aviso] muitas combinações: cada retreino treina o modelo do zero. "
              "Restrinja com --models / --presets / --detectors se demorar.\n")

    if args.initial_train_days is not None:
        janela_por_preset = {pn: args.initial_train_days for pn in preset_names}
        print(f"[manual] --initial-train-days {args.initial_train_days} para todos os presets")
    else:
        janela_por_preset = {pn: auto_initial_train_days(df_pre, args.equipment, pn)
                             for pn in preset_names}

    controle_fim = pd.Timestamp(args.controle_fim) if args.controle_fim else None
    drift_times = [pd.Timestamp(t) for t in (args.drift_time or [])]

    output_dir = Path(args.output_dir) / f"{args.equipment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_events = getattr(config, "failure_events", None) or (
        [config.failure_date] if getattr(config, "failure_date", None) else None)
    jan_main = janelas_prefalha(failure_events, args.prefailure_days)

    resumo_linhas: list[dict] = []
    n_erros = 0

    for model_name in model_names:
        for preset_name in preset_names:
            print(f"\n{'#' * 70}\n# modelo={model_name} | preset={preset_name}\n{'#' * 70}")
            p = Params(
                initial_train_days=janela_por_preset[preset_name], min_era_days=args.min_era_days,
                chunk_days=args.chunk_days, lookback_horas=args.lookback_horas,
                alarm_sigma=args.alarm_sigma, persist_k=args.persist_k, persist_n=args.persist_n,
                confirm_dias=args.confirm_dias, confirm_disparos=args.confirm_disparos,
                expira_suspeita_dias=args.expira_suspeita_dias,
                min_dias_regime_novo=args.min_dias_regime_novo,
                buffer_transicao_dias=args.buffer_transicao_dias,
                fator_nivel_normal=args.fator_nivel_normal,
                deg_patterns=tuple(args.deg_patterns), regime_cols=tuple(args.regime_cols),
                deg_window_days=args.deg_window_days, deg_slope_rel=args.deg_slope_rel,
                bloquear_degradacao=not args.permitir_retreino_em_degradacao,
                cooldown_bloqueio_dias=args.cooldown_bloqueio_dias,
                permitir_recalibracao=not args.sem_recalibracao, limite_leve=args.limite_leve,
                min_ganho_retreino=args.min_ganho_retreino,
                cooldown_rejeicao_dias=args.cooldown_rejeicao_dias,
                excluir_anomalias_treino=not args.manter_anomalias_no_treino,
                max_dias_episodio_excluir=args.max_dias_episodio_excluir,
                margem_anomalia_horas=args.margem_anomalia_horas,
                min_horas_politica=args.min_horas_politica, seed=args.seed,
                failure_events=tuple(failure_events or ()), prefailure_days=args.prefailure_days,
                gate_anomalia=not args.sem_gate_anomalia, gate_janela_dias=args.gate_janela_dias,
                gate_exceeds_pct=args.gate_exceeds_pct, teto_bloqueio_dias=args.teto_bloqueio_dias,
                quarentena_dias=args.quarentena_dias, min_dias_diagnostico=args.min_dias_diagnostico,
                regime_delta_min=args.regime_delta_min, resid_min=args.resid_min,
                tend_min=args.tend_min, extrap_resid_fator=args.extrap_resid_fator,
                gate_continuo=not args.sem_gate_continuo, persist_contexto_n=args.persist_contexto_n,
                limiar_robusto=args.limiar_robusto, limiar_mad=args.limiar_mad,
            )

            try:
                baseline = treinar_baseline(df_pre, args.equipment, model_name, preset_name,
                                            device, args.threshold_percentile, p, spd)
            except Exception as exc:
                print(f"  [ERRO] baseline {model_name}|{preset_name} falhou: {type(exc).__name__}: {exc}")
                n_erros += len(detector_names)
                continue

            bench = pd.DataFrame()
            if controle_fim is not None and drift_times:
                bench = benchmark_detectores(
                    df_pre, args.equipment, model_name, preset_name, device,
                    args.threshold_percentile, baseline, p, spd, detector_names,
                    controle_fim, drift_times, args.horizonte_dias)
                if len(bench):
                    bench.to_csv(output_dir / f"benchmark_{model_name}_{preset_name}.csv", index=False)
                    print("\n[benchmark] falsos alarmes × atraso por detector (baseline):")
                    print(bench.to_string(index=False))

            resultados_combo: dict[str, tuple] = {}

            try:
                fs_ctrl, m_ctrl, corte_ctrl = rodar_sem_adaptacao(
                    df_pre, args.equipment, model_name, preset_name, device,
                    args.threshold_percentile, baseline, p, config,
                    args.prefailure_days, args.normal_end_days)
                pref_c = f"{model_name}_{preset_name}_sem_adaptacao"
                fs_ctrl.to_parquet(output_dir / f"full_scores_{pref_c}.parquet")
                ep_ctrl = extrair_episodios(fs_ctrl, limiar_por_era={0: baseline.threshold})
                ep_ctrl.to_csv(output_dir / f"alertas_anomalia_{pref_c}.csv", index=False)
                lg_ctrl = pd.DataFrame([{"era": 0, "retrain_at": corte_ctrl, "motivo": "baseline_inicial",
                                         "threshold": baseline.threshold}])
                ev_vazio = pd.DataFrame(columns=["decisao", "data"])
                plotar_timeline(fs_ctrl, lg_ctrl, output_dir / f"timeline_{pref_c}.png",
                                failure_events=failure_events, eventos=ev_vazio,
                                title=f"{args.equipment} — {model_name}|{preset_name} sem adaptação (controle)")
                resumo_linhas.append({
                    "model": model_name, "preset": preset_name, "detector": "sem_adaptacao",
                    "n_retreinos": 0, "n_recalibracoes": 0, "n_bloqueios_degradacao": 0,
                    "n_adiamentos": 0, "n_retreinos_rejeitados": 0, "n_adaptacoes_prefalha": 0,
                    "n_episodios_anomalia": len(ep_ctrl), "dias_vermelho": np.nan,
                    "exceeds_pct": 100.0 * float(fs_ctrl["exceeds"].mean()),
                    "thr_rel_baseline": thr_rel(baseline.threshold, baseline.ref_errors),
                    "composite_score": m_ctrl.get("composite_score"),
                    "prefailure_alert_rate": m_ctrl.get("prefailure_alert_rate"),
                    "normal_alert_rate": m_ctrl.get("normal_alert_rate"),
                })
                resultados_combo["sem_adaptacao"] = (fs_ctrl, lg_ctrl, ev_vazio)
                print(f"\n[controle] sem adaptação: composite_score={m_ctrl.get('composite_score')} | "
                      f"{len(ep_ctrl)} episódio(s) de anomalia")
            except Exception as exc:
                print(f"  [ERRO] controle sem adaptação falhou: {type(exc).__name__}: {exc}")

            for nome_detector in detector_names:
                print(f"{'=' * 70}\n{model_name} | {preset_name} | {nome_detector}\n{'=' * 70}")
                try:
                    full_scores, retrain_log, drift_events, chunk_log = drift_triggered_walkforward(
                        df_pre, args.equipment, model_name, args.threshold_percentile,
                        nome_detector, device, baseline, p, preset=preset_name)
                except Exception as exc:
                    print(f"  [ERRO] {model_name}|{preset_name}|{nome_detector} falhou: "
                          f"{type(exc).__name__}: {exc}")
                    n_erros += 1
                    continue

                prefixo = f"{model_name}_{preset_name}_{nome_detector}"
                retrain_log.to_csv(output_dir / f"retrain_log_{prefixo}.csv", index=False)
                drift_events.to_csv(output_dir / f"drift_events_{prefixo}.csv", index=False)
                chunk_log.to_csv(output_dir / f"chunk_log_{prefixo}.csv", index=False)
                full_scores.to_parquet(output_dir / f"full_scores_{prefixo}.parquet")
                limiares = dict(zip(retrain_log["era"], retrain_log["threshold"]))
                episodios = extrair_episodios(full_scores, limiar_por_era=limiares)
                episodios.to_csv(output_dir / f"alertas_anomalia_{prefixo}.csv", index=False)
                plotar_timeline(full_scores, retrain_log, output_dir / f"timeline_{prefixo}.png",
                                failure_events=failure_events, eventos=drift_events,
                                title=f"{args.equipment} — {model_name}|{preset_name} + {nome_detector}")

                motivos = retrain_log["motivo"].astype(str)
                n_retreinos = int(motivos.str.startswith("drift_").sum())
                n_recal = int(motivos.str.startswith("recalibracao_").sum())
                metrics = pontuar_resultado(full_scores, config, args.prefailure_days,
                                            args.normal_end_days)
                pct_regime_novo = (100.0 * float((full_scores["contexto"] == "regime_novo").mean())
                                   if len(full_scores) and "contexto" in full_scores else np.nan)
                resumo_linhas.append({
                    "model": model_name, "preset": preset_name, "detector": nome_detector,
                    "n_retreinos": n_retreinos, "n_recalibracoes": n_recal,
                    "n_bloqueios_degradacao": _conta(drift_events, "decisao", "retreino_bloqueado_degradacao")
                                              + _conta(drift_events, "decisao", "bloqueio_escalado_operacao"),
                    "n_adiamentos": _conta(drift_events, "decisao", "retreino_adiado_regime_curto")
                                    + _conta(drift_events, "decisao", "retreino_adiado_dados_insuficientes"),
                    "n_retreinos_rejeitados": _conta(drift_events, "decisao", "retreino_rejeitado_sem_ganho"),
                    "n_alarmes_falsos_descartados": _conta(drift_events, "decisao", "alarme_falso_descartado"),
                    "n_adaptacoes_prefalha": adaptacoes_em_prefalha(retrain_log, jan_main),
                    "max_treino_frac_anomalia": _max_col(retrain_log, "treino_frac_anomalia"),
                    "max_treino_frac_prefalha": _max_col(retrain_log, "treino_frac_prefalha"),
                    "max_thr_x_baseline": _max_col(retrain_log, "thr_x_baseline"),
                    "n_episodios_anomalia": len(episodios),
                    "exceeds_pct": 100.0 * float(full_scores["exceeds"].mean()) if len(full_scores) else np.nan,
                    "thr_rel_baseline": thr_rel(baseline.threshold, baseline.ref_errors),
                    "dias_vermelho": int((chunk_log["status"] == "vermelho").sum()) if len(chunk_log) else 0,
                    "composite_score": metrics.get("composite_score"),
                    "prefailure_alert_rate": metrics.get("prefailure_alert_rate"),
                    "normal_alert_rate": metrics.get("normal_alert_rate"),
                    "pct_tempo_regime_novo": pct_regime_novo,
                })
                resultados_combo[nome_detector] = (full_scores, retrain_log, drift_events)
                print(f"  {n_retreinos} retreino(s) | {n_recal} recalibração(ões) | "
                      f"{len(episodios)} episódio(s) de anomalia | "
                      f"composite_score={metrics.get('composite_score')}")
                pd.DataFrame(resumo_linhas).to_csv(output_dir / "resumo_parcial.csv", index=False)

            ctrl = next((l for l in resumo_linhas if l["model"] == model_name
                         and l["preset"] == preset_name and l["detector"] == "sem_adaptacao"), None)
            if ctrl and ctrl.get("composite_score") is not None:
                for l in resumo_linhas:
                    if (l["model"], l["preset"]) == (model_name, preset_name) \
                            and l["detector"] != "sem_adaptacao" and l.get("composite_score") is not None:
                        l["delta_composite_vs_controle"] = l["composite_score"] - ctrl["composite_score"]

            if resultados_combo:
                plotar_comparativo(resultados_combo,
                                   output_dir / f"timeline_comparativa_{model_name}_{preset_name}.png",
                                   failure_events=failure_events,
                                   title=f"{args.equipment} — {model_name}|{preset_name}: comparação de técnicas")

            if len(bench):
                for linha in resumo_linhas:
                    if linha["model"] == model_name and linha["preset"] == preset_name:
                        b = bench[bench["detector"] == linha["detector"]]
                        if len(b):
                            linha.update(b.drop(columns="detector").iloc[0].to_dict())

    if not resumo_linhas:
        raise RuntimeError("Nenhuma combinação rodou com sucesso.")

    resumo = pd.DataFrame(resumo_linhas)
    if args.max_fp_rate and args.max_fp_rate > 0:
        resumo["aprovado"] = resumo["normal_alert_rate"].fillna(1.0) <= args.max_fp_rate
        print(f"\n[constraint] max_fp_rate={args.max_fp_rate:.2%} → "
              f"{int(resumo['aprovado'].sum())}/{len(resumo)} combinação(ões) dentro do teto")
        resumo = resumo.sort_values(["aprovado", "composite_score"], ascending=[False, False],
                                    na_position="last")
    else:
        resumo = resumo.sort_values("composite_score", ascending=False, na_position="last")

    resumo.to_csv(output_dir / "resumo_tecnicas.csv", index=False)
    print(f"\n{'=' * 70}\nRESUMO — TOP 15 COMBINAÇÕES (inclui a linha 'sem_adaptacao' de controle)\n{'=' * 70}")
    print(resumo.head(15).to_string(index=False))
    print(f"\n{len(resumo)} combinações | {n_erros} falharam | resumo em resumo_tecnicas.csv")
    print(f"\n✓ Resultados salvos em: {output_dir.resolve()}")

    if task is not None and not args.no_clearml_upload:
        print("\nUpload ao ClearML...")
        task.upload_artifact("resumo_tecnicas", artifact_object=resumo)
        logger = task.get_logger()
        for _, row in resumo.head(20).iterrows():
            rotulo = f"{row['model']}|{row['preset']}|{row['detector']}"
            if pd.notna(row.get("composite_score")):
                logger.report_scalar("drift/composite_score", rotulo, float(row["composite_score"]), 0)
            logger.report_scalar("drift/n_retreinos", rotulo, float(row["n_retreinos"]), 0)
            logger.report_scalar("drift/n_recalibracoes", rotulo, float(row["n_recalibracoes"]), 0)
        for _, row in resumo.head(5).iterrows():
            prefixo = f"{row['model']}_{row['preset']}_{row['detector']}"
            img = output_dir / f"timeline_{prefixo}.png"
            if img.exists():
                logger.report_image("timelines_top5", prefixo, local_path=str(img), iteration=0)
        logger.report_table("resumo", "grade_completa", table_plot=resumo)

        import io, json

        def _pq(df: pd.DataFrame) -> bytes:
            buf = io.BytesIO()
            df.to_parquet(buf)
            return buf.getvalue()

        series, logs_all, events_all, chunks_all, alerts_all = {}, [], [], [], []
        for _, row in resumo.iterrows():
            prefixo = f"{row['model']}_{row['preset']}_{row['detector']}"
            combo = f"{row['model']} | {row['preset']} | {row['detector']}"
            fs_path = output_dir / f"full_scores_{prefixo}.parquet"
            if fs_path.exists():
                fs = pd.read_parquet(fs_path)
                fs.index.name = "timestamp"
                series[combo] = _pq(fs.reset_index())
            for pref, lista in (("retrain_log", logs_all), ("drift_events", events_all),
                                ("chunk_log", chunks_all), ("alertas_anomalia", alerts_all)):
                path = output_dir / f"{pref}_{prefixo}.csv"
                if path.exists() and path.stat().st_size > 5:
                    lista.append(pd.read_csv(path).assign(combo=combo))

        cat = lambda l: _pq(pd.concat(l, ignore_index=True)) if l else None
        pacote = {
            "versao": 2,
            "meta": {"equipment": args.equipment,
                     "failure_events": [str(e) for e in (failure_events or [])],
                     "prefailure_days": args.prefailure_days, "normal_end_days": args.normal_end_days,
                     "args": json.loads(json.dumps(vars(args), default=str))},
            "resumo": _pq(resumo), "series": series,
            "logs": cat(logs_all), "eventos": cat(events_all),
            "chunks": cat(chunks_all), "alertas_anomalia": cat(alerts_all),
        }
        task.upload_artifact("resultados", artifact_object=[pacote], auto_pickle=True,
                             wait_on_upload=True)
        print(f"✓ Upload completo — artefato 'resultados' v2 ({len(series)} séries)")


if __name__ == "__main__":
    main()