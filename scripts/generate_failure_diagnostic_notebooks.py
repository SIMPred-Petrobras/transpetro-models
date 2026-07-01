"""
Gera notebooks de diagnostico para equipamentos com resultado fraco/incerto.

Os notebooks sao deliberadamente exploratorios: eles nao treinam modelo pesado.
Medem qualidade/cobertura dos dados, pobreza de operacao ligada, sensores mortos,
separabilidade normal vs pre-falha e um teto simples de deteccao a 1% FP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


EQUIPMENTS = {
    "B-4064A": {
        "config": "B-4064A-prod",
        "failure": "2024-08-30 07:58",
        "status": "re-baselinar por mudanca de regime termico em 2025",
        "out": "notebooks/b4064a/diagnostico_regime_deploy.ipynb",
        "fallbacks": ["Dados-novos/B-4064A_novos.feather"],
    },
    "B-0302C": {
        "config": "B-0302C",
        "failure": "2024-08-30",
        "status": "fraco: equipamento majoritariamente desligado e sensores de motor sem sinal util",
        "out": "notebooks/b0302c/diagnostico_modelo_fraco.ipynb",
        "fallbacks": ["DadosV2/B-0302C_pivoted.feather", "DadosV2/B-0302C.feather"],
    },
    "B-4703.24001B": {
        "config": "B-4703.24001B",
        "failure": "2022-10-01",
        "status": "fraco: desgaste de rolamento/mancal pouco visivel em RMS/telemetria baixa frequencia",
        "out": "notebooks/b4703_24001b/diagnostico_modelo_fraco.ipynb",
        "fallbacks": ["DadosV2/B-4703.24001B.feather"],
    },
    "B-90001A": {
        "config": "B-90001A_interpolated",
        "failure": "2021-08-28",
        "status": "nao trabalhado a fundo; base interpolada nao encontrada localmente",
        "out": "notebooks/b-90001a_interpolated/diagnostico_modelo_fraco.ipynb",
        "fallbacks": ["Dados/B-90001A_interpolated.csv", "Dados/B-90001A.feather"],
    },
    "B-3403C": {
        "config": "B-3403C_interpolated",
        "failure": "2023-09-12",
        "status": "base interpolada da Lara; arquivo interpolado nao encontrado localmente",
        "out": "notebooks/b-3403c_interpolated/diagnostico_modelo_fraco.ipynb",
        "fallbacks": ["Dados/B-3403C_interpolated.csv", "DadosV2/B-3403C_pivoted.feather", "DadosV2/B-3403C.feather"],
    },
    "B-402E": {
        "config": "B-402E",
        "failure": "2019-10-30 11:06",
        "status": "descartado: falha catastrofica/subita com instrumentacao insuficiente",
        "out": "notebooks/b402e/diagnostico_modelo_fraco.ipynb",
        "fallbacks": ["Dados/B-402E.feather"],
    },
    "B-5401A": {
        "config": "B-5401A",
        "failure": "2024-08-10",
        "status": "descartado: curto de motor sem precursor nos sensores disponiveis",
        "out": "notebooks/b5401a/diagnostico_modelo_fraco.ipynb",
        "fallbacks": ["DadosV2/B-5401A_pivoted.feather", "DadosV2/B-5401A.feather"],
    },
}


COMMON_CODE = r'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(ROOT / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_rows", 80)
pd.set_option("display.max_columns", 80)
'''


ANALYSIS_CODE = r'''
EQUIPMENT = "__EQUIPMENT__"
CONFIG_KEY = "__CONFIG__"
FAILURE_DATE = pd.Timestamp("__FAILURE__")
STATUS_ATUAL = "__STATUS__"
FALLBACK_FILES = __FALLBACKS__

def _read_any(path):
    path = Path(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)

def _ensure_datetime_index(df, datetime_column=None):
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    candidates = []
    if datetime_column:
        candidates.append(datetime_column)
    candidates.extend(["datetime", "Timestamp", "Data Hora", "data_hora", "DataHora"])
    for col in candidates:
        if col in df.columns:
            out = df.set_index(col)
            out.index = pd.to_datetime(out.index)
            return out.sort_index()
    first = df.columns[0]
    parsed = pd.to_datetime(df[first], errors="coerce")
    if parsed.notna().mean() > 0.8:
        out = df.set_index(first)
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
    raise ValueError("Nao encontrei coluna temporal clara.")

def load_data():
    cfg = EQUIPMENT_CONFIGS.get(CONFIG_KEY)
    try:
        df = load_equipment_data(CONFIG_KEY, from_clearml=False)
        return df, f"config:{CONFIG_KEY}"
    except Exception as exc:
        print(f"load_equipment_data falhou para {CONFIG_KEY}: {type(exc).__name__}: {exc}")
    for rel in FALLBACK_FILES:
        path = ROOT / rel
        if path.exists():
            df = _read_any(path)
            dt_col = cfg.datetime_column if cfg else None
            return _ensure_datetime_index(df, dt_col), str(rel)
    raise FileNotFoundError(f"Nenhum fallback encontrado: {FALLBACK_FILES}")

def numeric_live(df):
    num = df.select_dtypes(include=[np.number]).copy()
    rows = []
    for col in num.columns:
        s = num[col]
        nn = int(s.notna().sum())
        zero_frac = float((s.fillna(0) == 0).mean()) if len(s) else 0.0
        rows.append({
            "sensor": col,
            "n_validos": nn,
            "missing_pct": float(s.isna().mean()),
            "zero_pct": zero_frac,
            "std": float(s.std(skipna=True)) if nn else np.nan,
            "min": float(s.min(skipna=True)) if nn else np.nan,
            "p50": float(s.median(skipna=True)) if nn else np.nan,
            "max": float(s.max(skipna=True)) if nn else np.nan,
        })
    q = pd.DataFrame(rows)
    dead_mask = (q["n_validos"] == 0) | (q["std"].fillna(0) == 0) | (q["zero_pct"] >= 0.995)
    dead = q.loc[dead_mask, "sensor"].tolist()
    live = num.drop(columns=dead, errors="ignore")
    return num, live, q.sort_values(["zero_pct", "missing_pct"], ascending=False), dead

def pick_activity(live):
    preferred = ("corrente", "current", "vaz", "flow", "pressao descarga", "pressão descarga")
    for kw in preferred:
        cand = [c for c in live.columns if kw in c.lower()]
        if cand:
            return max(cand, key=lambda c: live[c].quantile(0.99) - live[c].quantile(0.01))
    spread = live.quantile(0.99) - live.quantile(0.01)
    return spread.idxmax()

def activity_cutoff(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 200 or x.nunique() < 10:
        return None, None, False, None
    arr = x.to_numpy().reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=0).fit(arr)
    g2 = GaussianMixture(2, random_state=0).fit(arr)
    means = np.sort(g2.means_.ravel())
    stds = np.sqrt(g2.covariances_.ravel())
    gap = abs(means[1] - means[0]) / max(float(np.max(stds)), 1e-9)
    bimodal = (g2.bic(arr) + 10 < g1.bic(arr)) and gap > 2.0
    if not bimodal:
        return None, means, False, gap
    grid = np.linspace(means[0], means[1], 500).reshape(-1, 1)
    cutoff = float(grid[np.argmin(np.exp(g2.score_samples(grid)))][0])
    off_frac = float((x < cutoff).mean())
    return cutoff, means, off_frac, gap

def robust_z(frame, normal):
    med = normal.median()
    mad = (normal - med).abs().median().replace(0, np.nan)
    scale = 1.4826 * mad
    return (frame - med) / scale

raw, source = load_data()
cfg = EQUIPMENT_CONFIGS.get(CONFIG_KEY)
print(f"{EQUIPMENT} | status: {STATUS_ATUAL}")
print(f"Fonte usada: {source}")
print(f"Periodo bruto: {raw.index.min()} -> {raw.index.max()} | linhas={len(raw):,} | colunas={raw.shape[1]}")
print(f"Falha conhecida: {FAILURE_DATE}")
if not (raw.index.min() <= FAILURE_DATE <= raw.index.max()):
    print("ATENCAO: data da falha esta fora do periodo do arquivo carregado.")

num, live, quality, dead = numeric_live(raw)
print(f"Sensores numericos: {num.shape[1]} | vivos: {live.shape[1]} | mortos/quase mortos: {len(dead)}")
if dead:
    print("Mortos/quase mortos:", dead)
display(quality.head(30).style.format({
    "missing_pct": "{:.1%}", "zero_pct": "{:.1%}", "std": "{:.4g}",
    "min": "{:.4g}", "p50": "{:.4g}", "max": "{:.4g}",
}))

if cfg is not None and getattr(cfg, "pre_split_steps", None):
    try:
        proc_pre, _, report = run_preprocessing(
            raw.copy(), cfg.pre_split_steps, return_artifacts=True, return_report=True
        )
        print(
            f"Depois do pre_split_steps da config: {len(proc_pre):,} linhas "
            f"({len(proc_pre) / max(len(raw), 1):.1%} do bruto), colunas={proc_pre.shape[1]}"
        )
    except Exception as exc:
        proc_pre = None
        print(f"Nao consegui aplicar pre_split_steps da config: {type(exc).__name__}: {exc}")
else:
    proc_pre = None
    print("Config sem pre_split_steps; analise segue no bruto numerico.")
'''


PLOTS_AND_METRICS = r'''
base = proc_pre.select_dtypes(include=[np.number]) if proc_pre is not None else live
_, live2, quality2, dead2 = numeric_live(base)
if live2.empty:
    raise RuntimeError("Sem sensores vivos para diagnostico.")

act = pick_activity(live2)
cutoff, modes, off_frac, gap = activity_cutoff(live2[act])
print(f"Sensor de atividade escolhido: {act}")
if cutoff is not None:
    print(f"Bimodal ligado/desligado: modos={modes[0]:.4g}/{modes[1]:.4g}, gap={gap:.2f}, cutoff={cutoff:.4g}, desligado={off_frac:.1%}")
    operating = live2[live2[act] >= cutoff]
else:
    print("Sem bimodalidade clara para ligado/desligado; usando todos os pontos vivos.")
    operating = live2.copy()
print(f"Amostras em operacao usadas na separabilidade: {len(operating):,} ({len(operating)/max(len(live2), 1):.1%} dos pontos vivos)")

fig, ax = plt.subplots(figsize=(12, 3.5))
x = live2[act].dropna()
ax.hist(x, bins=120, color="steelblue", alpha=0.7)
if cutoff is not None:
    ax.axvline(cutoff, color="crimson", linestyle="--", label=f"cutoff={cutoff:.3g}")
ax.set_title(f"{EQUIPMENT} - histograma do sensor de atividade")
ax.set_xlabel(act)
ax.set_ylabel("contagem")
ax.legend()
plt.show()

spread = (operating.quantile(0.99) - operating.quantile(0.01)).sort_values(ascending=False)
cols = list(spread.head(min(24, len(spread))).index)
ncol = 3
nrow = int(np.ceil(len(cols) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, max(3, 2.5 * nrow)))
axes = np.atleast_1d(axes).ravel()
for ax, col in zip(axes, cols):
    ax.hist(operating[col].dropna(), bins=80, color="steelblue", alpha=0.8)
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[len(cols):]:
    ax.axis("off")
fig.suptitle(f"{EQUIPMENT} - histogramas dos sensores vivos em operacao", y=1.002)
fig.tight_layout()
plt.show()

normal_end_days = getattr(cfg, "normal_end_days", None) or 20
prefailure_days = getattr(cfg, "prefailure_days", None) or 7
normal_end = FAILURE_DATE - pd.Timedelta(days=normal_end_days)
pre_start = FAILURE_DATE - pd.Timedelta(days=prefailure_days)
normal = operating[operating.index < normal_end]
pre = operating[(operating.index >= pre_start) & (operating.index < FAILURE_DATE)]
print(f"Janela normal: < {normal_end} | n={len(normal):,}")
print(f"Janela pre-falha: {pre_start} -> {FAILURE_DATE} | n={len(pre):,}")

if len(normal) >= 50 and len(pre) >= 10:
    z_normal = robust_z(normal, normal).abs()
    z_pre = robust_z(pre, normal).abs()
    thr = z_normal.quantile(0.99)
    det = (z_pre > thr).mean().sort_values(ascending=False)
    shift = ((pre.median() - normal.median()).abs() / ((normal - normal.median()).abs().median().replace(0, np.nan) * 1.4826)).sort_values(ascending=False)
    sep = pd.DataFrame({
        "detec_pre_falha_1pct_fp": det,
        "deslocamento_robusto_sigma": shift,
        "normal_p50": normal.median(),
        "pre_falha_p50": pre.median(),
        "normal_p99": normal.quantile(0.99),
        "pre_falha_max": pre.max(),
    }).sort_values(["detec_pre_falha_1pct_fp", "deslocamento_robusto_sigma"], ascending=False)
    display(sep.head(20).style.format({
        "detec_pre_falha_1pct_fp": "{:.1%}",
        "deslocamento_robusto_sigma": "{:.2f}",
        "normal_p50": "{:.4g}",
        "pre_falha_p50": "{:.4g}",
        "normal_p99": "{:.4g}",
        "pre_falha_max": "{:.4g}",
    }))
    print(f"Melhor sensor isolado @1% FP: {sep.index[0]} -> deteccao {sep.iloc[0]['detec_pre_falha_1pct_fp']:.1%}")

    top = list(sep.head(min(6, len(sep))).index)
    fig, axes = plt.subplots(len(top), 1, figsize=(14, max(3, 2.0 * len(top))), sharex=True)
    axes = np.atleast_1d(axes)
    window = operating.loc[FAILURE_DATE - pd.Timedelta(days=45): FAILURE_DATE]
    for ax, col in zip(axes, top):
        ax.plot(window.index, window[col], lw=1.0)
        ax.axvline(normal_end, color="gray", linestyle="--", lw=1, label="fim normal")
        ax.axvline(FAILURE_DATE, color="crimson", linestyle="--", lw=1, label="falha")
        ax.set_ylabel(col, fontsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{EQUIPMENT} - 45 dias antes da falha nos sensores mais separaveis", y=1.002)
    fig.tight_layout()
    plt.show()

    cols_model = sep.head(min(12, len(sep))).index.tolist()
    train = normal[cols_model].dropna()
    pre_m = pre[cols_model].dropna()
    if len(train) >= 50 and len(pre_m) >= 10:
        model = LedoitWolf().fit(train)
        normal_score = pd.Series(model.mahalanobis(train), index=train.index)
        pre_score = pd.Series(model.mahalanobis(pre_m), index=pre_m.index)
        mthr = float(normal_score.quantile(0.99))
        mdet = float((pre_score > mthr).mean())
        print(f"Proxy multivariado Mahalanobis @1% FP: deteccao pre-falha={mdet:.1%} | threshold={mthr:.3g}")
else:
    print("Janelas normal/pre-falha pequenas demais para estimar separabilidade de forma honesta.")

if len(operating):
    year_counts = operating.groupby(operating.index.year).size()
    print("Pontos em operacao por ano:")
    display(year_counts.to_frame("n_pontos"))
    yearly = operating.groupby(operating.index.year).median()
    if len(yearly) > 1:
        ref = yearly.iloc[0]
        delta = (yearly - ref).abs().max().sort_values(ascending=False)
        print("Maiores deslocamentos absolutos de mediana entre anos:")
        display(delta.head(15).to_frame("max_delta_mediana"))
'''


CONCLUSION = r'''
## Leitura operacional

Use a tabela de separabilidade como um "teto" antes de treinar AutoML:

- `detec_pre_falha_1pct_fp` baixo indica que, mantendo falso positivo em torno de 1%, o sinal da
  falha quase nunca sai da distribuicao normal.
- Muitos sensores mortos/quase zerados indicam que o modo de falha nao esta coberto pela
  instrumentacao disponivel.
- Percentual alto de desligado reduz a quantidade de dados de operacao real; o modelo aprende
  muito estado parado e pouco regime saudavel.
- Deslocamento forte por ano/regime indica que o problema e re-baseline, nao hiperparametro.

Para este equipamento, o proximo passo deve ser decidido pela causa dominante acima: filtrar
estado desligado, re-baselinar no regime atual, buscar sensores do subsistema que falhou ou
descartar o caso para modelo preditivo com a instrumentacao atual.
'''


def make_notebook(equipment: str, meta: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    title = f"# {equipment} - diagnostico do resultado fraco/incerto\n\n"
    title += f"**Status atual:** {meta['status']}.\n\n"
    title += (
        "Objetivo: documentar por que o modelo tende a render pouco antes de gastar mais tuning. "
        "A analise mede sensores mortos, pobreza de dados em operacao, bimodalidade ligado/desligado, "
        "mudanca de regime e separabilidade normal vs pre-falha com uma meta simples de 1% FP."
    )
    analysis_code = (
        ANALYSIS_CODE
        .replace("__EQUIPMENT__", equipment)
        .replace("__CONFIG__", meta["config"])
        .replace("__FAILURE__", meta["failure"])
        .replace("__STATUS__", meta["status"])
        .replace("__FALLBACKS__", repr(meta["fallbacks"]))
    )
    nb.cells = [
        nbf.v4.new_markdown_cell(title),
        nbf.v4.new_code_cell(COMMON_CODE),
        nbf.v4.new_code_cell(analysis_code),
        nbf.v4.new_code_cell(PLOTS_AND_METRICS),
        nbf.v4.new_markdown_cell(CONCLUSION),
    ]
    return nb


def generate(selected: list[str] | None = None) -> list[Path]:
    keys = selected or list(EQUIPMENTS)
    written = []
    for equipment in keys:
        meta = EQUIPMENTS[equipment]
        out = ROOT / meta["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        nb = make_notebook(equipment, meta)
        nbf.write(nb, out)
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equipment", choices=list(EQUIPMENTS), action="append")
    args = parser.parse_args()
    for path in generate(args.equipment):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
