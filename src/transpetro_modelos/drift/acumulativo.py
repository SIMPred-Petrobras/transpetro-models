"""
Retreino acumulativo depois de uma mudança de conceito (modelo provisório que amadurece).

Etapa k (k = 1..n): treina com os k primeiros meses do novo normal e opera durante o mês seguinte.
A última etapa (k = n, 12 meses) fica fixa até o fim dos dados. Mesma receita do retreino de
produção: VAE com a arquitetura aprovada no replay, limiar média + y·σ do erro na janela de treino
e persistência k de n leituras.

Para cada etapa mede, sobre o mês em que aquele modelo estaria em produção:
  - alarme % (sem falhas conhecidas nesse período: todo alarme é candidato a falso positivo,
    exceto os episódios reais já justificados);
  - falha sintética (assinatura real de 2022) injetada nesse mês, a 100 % e 50 %;
  - falha real de 2022 (série que nenhuma etapa viu).

`serie_emendada()` junta, mês a mês, os alarmes do modelo que estaria em produção em cada momento
e `episodios()` lista os episódios com o sensor que mais contribuiu para o erro.

Os modelos treinados ficam em cache em `saida/etapa_XX/`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
from transpetro_modelos.data.loading import load_equipment_data
from transpetro_modelos.data.preprocessing import run_preprocessing
from transpetro_modelos.training.automl import build_model
from transpetro_modelos.training.train import train_vae
from transpetro_modelos.training.evaluate import compute_vae_errors
from transpetro_modelos.drift import battery

ARQUITETURA = {"camadas": (128, 64, 32), "latente": 16, "lr": 1e-4, "semente": 1}   # aprovada no replay
Y_ALARME, K_PERS, N_PERS = 6.5, 15, 20
FALHA_2022, RAMPA_2022 = pd.Timestamp("2022-07-06 10:00"), pd.Timestamp("2022-06-29")


def kofn(flags: pd.Series, k: int = K_PERS, n: int = N_PERS) -> pd.Series:
    return (flags.astype(int).rolling(n, min_periods=n).sum() >= k).fillna(False)


def etapas(inicio: str, n_meses: int, fim_dados: pd.Timestamp) -> list[dict]:
    ini = pd.Timestamp(inicio)
    out = []
    for k in range(1, n_meses + 1):
        treino_fim = ini + pd.DateOffset(months=k)
        vivo_fim = fim_dados if k == n_meses else treino_fim + pd.DateOffset(months=1)
        out.append({"etapa": k, "treino_ini": ini, "treino_fim": treino_fim,
                    "vivo_ini": treino_fim, "vivo_fim": vivo_fim})
    return out


def _erros(model, X: pd.DataFrame) -> pd.Series:
    torch.manual_seed(0)
    return pd.Series(compute_vae_errors(model, X), index=X.index)


def erros_por_sensor(model, X: pd.DataFrame) -> pd.DataFrame:
    """Erro quadrático de reconstrução de cada sensor (mesmas unidades normalizadas do treino)."""
    torch.manual_seed(0)
    model.eval()
    partes = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            lote = torch.tensor(X.values[i:i + 4096], dtype=torch.float32)
            recon = model(lote)[0]
            partes.append(((recon - lote) ** 2).numpy())
    return pd.DataFrame(np.vstack(partes), index=X.index, columns=X.columns)


def _treinar_etapa(pre: pd.DataFrame, preset: list, et: dict, pasta: Path, epochs: int):
    pasta.mkdir(parents=True, exist_ok=True)
    treino = pre[(pre.index >= et["treino_ini"]) & (pre.index < et["treino_fim"])]
    n_val = int(len(treino) * 0.15)
    fit, va = treino.iloc[:-n_val], treino.iloc[-n_val:]
    Xtr, art, _ = run_preprocessing(fit, preset, return_artifacts=True, return_report=True)
    Xva = run_preprocessing(va, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    a = ARQUITETURA
    m = build_model("vae", Xtr.shape[1], dense_layers=list(a["camadas"]), latent_dim=a["latente"])
    if (pasta / "model.pt").exists():
        m.load_state_dict(torch.load(pasta / "model.pt", map_location="cpu"))
        art = pickle.load(open(pasta / "art.pkl", "rb"))
    else:
        torch.manual_seed(a["semente"]); np.random.seed(a["semente"])
        dl = lambda X, sh: DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float32)), batch_size=256, shuffle=sh)
        m = train_vae(m, dl(Xtr, True), dl(Xva, False), epochs=epochs, learning_rate=a["lr"], weight_decay=1e-5, patience=10)
        torch.save(m.state_dict(), pasta / "model.pt"); pickle.dump(art, open(pasta / "art.pkl", "wb"))
    m.eval()
    Xtreino = run_preprocessing(treino, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    e_tr = _erros(m, Xtreino)
    return m, art, float(e_tr.mean()), float(e_tr.std()), len(treino)


def _lead_sintetica(model, art, raw, cfg, preset, thr, t0, escala) -> float | None:
    sy = battery.BATTERY["B-8802B-2025"]["synthetic"]
    tf = t0 + pd.Timedelta(hours=sy["ramp_h"])
    trecho = raw[(raw.index >= t0 - pd.Timedelta(days=3)) & (raw.index <= tf + pd.Timedelta(hours=sy["hold_h"] + 12))]
    inj = battery.inject(trecho, sy["signature"], t0, sy["ramp_h"], sy["hold_h"], escala)
    pre, _, _ = run_preprocessing(inj, cfg.pre_split_steps)
    X = run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    fl = kofn(_erros(model, X) > thr)
    w = fl[(fl.index >= t0) & (fl.index <= tf + pd.Timedelta(hours=sy["hold_h"]))]
    return (tf - w[w].index.min()).total_seconds() / 3600 if w.any() else None


def _t0_sintetica(raw: pd.DataFrame, vivo_ini, vivo_fim) -> pd.Timestamp | None:
    """1º dia (10, 17 ou 24 do mês de operação) com a bomba operando ≥ 80 % do tempo nas 72 h da injeção."""
    for dia in (9, 16, 23):
        t0 = pd.Timestamp(vivo_ini) + pd.Timedelta(days=dia)
        if t0 + pd.Timedelta(hours=72) > pd.Timestamp(vivo_fim):
            continue
        w = raw.loc[t0: t0 + pd.Timedelta(hours=72), "Pressão Descarga"]
        if len(w) and (w > 35).mean() >= 0.8:
            return t0
    return None


def rodar(saida: str | Path, inicio: str = "2025-01-01", n_meses: int = 12, epochs: int = 60, log=print) -> pd.DataFrame:
    saida = Path(saida); saida.mkdir(parents=True, exist_ok=True)
    cfg = EQUIPMENT_CONFIGS["B-8802B-2025"]
    preset = get_preprocessing_steps("B-8802B-2025", "baseline")
    raw = load_equipment_data("B-8802B-2025", from_clearml=False)
    pre, _, _ = run_preprocessing(raw, cfg.pre_split_steps)
    raw22 = load_equipment_data("B-8802B", from_clearml=False)
    pre22, _, _ = run_preprocessing(raw22, cfg.pre_split_steps)

    linhas = []
    for et in etapas(inicio, n_meses, pre.index.max()):
        k = et["etapa"]
        m, art, mu, sd, n_tr = _treinar_etapa(pre, preset, et, saida / f"etapa_{k:02d}", epochs)
        thr = mu + Y_ALARME * sd
        X = run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
        e = _erros(m, X); fl = kofn(e > thr)
        vivo = (fl.index >= et["vivo_ini"]) & (fl.index < et["vivo_fim"])
        pd.DataFrame({"erro": e[vivo], "limiar": thr, "alarme": fl[vivo]}).to_parquet(saida / f"etapa_{k:02d}" / "vivo.parquet")
        erros_por_sensor(m, X[vivo]).to_parquet(saida / f"etapa_{k:02d}" / "vivo_sensores.parquet")

        X22 = run_preprocessing(pre22, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
        f22 = kofn(_erros(m, X22) > thr)
        rampa = f22[(f22.index >= RAMPA_2022) & (f22.index < FALHA_2022)]
        lead22 = (FALHA_2022 - rampa[rampa].index.min()).total_seconds() / 86400 if rampa.any() else None

        t0 = _t0_sintetica(raw, et["vivo_ini"], et["vivo_fim"])
        s100 = _lead_sintetica(m, art, raw, cfg, preset, thr, t0, 1.0) if t0 is not None else None
        s50 = _lead_sintetica(m, art, raw, cfg, preset, thr, t0, 0.5) if t0 is not None else None
        linha = {"etapa": k, "meses_de_treino": k, "treino_ini": et["treino_ini"], "treino_fim": et["treino_fim"],
                 "vivo_ini": et["vivo_ini"], "vivo_fim": et["vivo_fim"], "n_treino": n_tr, "limiar": thr,
                 "alarme_pct_vivo": 100 * float(fl[vivo].mean()) if vivo.any() else None,
                 "normal_2022_pct": 100 * float(f22[f22.index < RAMPA_2022].mean()), "falha_2022_dias": lead22,
                 "sintetica_em": t0, "sintetica_100_h": s100, "sintetica_50_h": s50}
        linhas.append(linha)
        log(f"[etapa {k:2d}] treino {k:2d} mês(es) → opera {et['vivo_ini']:%m/%Y}"
            f"{' em diante' if k == n_meses else ''} | alarme {linha['alarme_pct_vivo']:.3f} %"
            f" | falha 2022 {lead22 if lead22 is None else round(lead22, 2)} d"
            f" | sintética {s100 if s100 is None else round(s100)} h / {s50 if s50 is None else round(s50)} h")
    res = pd.DataFrame(linhas)
    res.to_csv(saida / "etapas.csv", index=False)
    return res


def sintetica_data_fixa(saida: str | Path, t0: str = "2026-06-10", n_meses: int = 12) -> pd.DataFrame:
    """Falha sintética injetada na MESMA data para todas as etapas: separa o efeito do modelo
    (quantos meses de treino) do efeito do mês em que a falha foi injetada."""
    saida = Path(saida)
    cfg = EQUIPMENT_CONFIGS["B-8802B-2025"]
    preset = get_preprocessing_steps("B-8802B-2025", "baseline")
    raw = load_equipment_data("B-8802B-2025", from_clearml=False)
    pre, _, _ = run_preprocessing(raw, cfg.pre_split_steps)
    t0 = pd.Timestamp(t0); linhas = []
    for et in etapas("2025-01-01", n_meses, pre.index.max()):
        m, art, mu, sd, _ = _treinar_etapa(pre, preset, et, saida / f"etapa_{et['etapa']:02d}", epochs=60)
        thr = mu + Y_ALARME * sd
        linhas.append({"etapa": et["etapa"],
                       "sintetica_fixa_100_h": _lead_sintetica(m, art, raw, cfg, preset, thr, t0, 1.0),
                       "sintetica_fixa_50_h": _lead_sintetica(m, art, raw, cfg, preset, thr, t0, 0.5)})
    return pd.DataFrame(linhas)


def serie_emendada(saida: str | Path, alarmes_quarentena: pd.DataFrame | None = None) -> pd.DataFrame:
    """Alarmes do modelo que estaria em produção em cada momento (coluna `etapa`; 0 = quarentena)."""
    saida = Path(saida); partes = []
    if alarmes_quarentena is not None:
        partes.append(alarmes_quarentena.assign(etapa=0))
    for p in sorted(saida.glob("etapa_*/vivo.parquet")):
        k = int(p.parent.name.split("_")[1])
        partes.append(pd.read_parquet(p).assign(etapa=k))
    return pd.concat(partes).sort_index()


def episodios(serie: pd.DataFrame, saida: str | Path, gap_h: float = 12) -> pd.DataFrame:
    """Episódios de alarme (fora da quarentena) com duração e o sensor que mais pesou no erro."""
    saida = Path(saida)
    al = serie[(serie["alarme"]) & (serie["etapa"] > 0)]
    if al.empty:
        return pd.DataFrame()
    novo = al.index.to_series().diff() > pd.Timedelta(hours=gap_h)
    grupo = novo.cumsum()
    sens = {k: pd.read_parquet(saida / f"etapa_{k:02d}" / "vivo_sensores.parquet") for k in al["etapa"].unique()}
    linhas = []
    for _, g in al.groupby(grupo):
        k = int(g["etapa"].iloc[0])
        contrib = sens[k].loc[g.index].mean()
        linhas.append({"inicio": g.index.min(), "fim": g.index.max(),
                       "duracao_h": round((g.index.max() - g.index.min()).total_seconds() / 3600 + 5 / 60, 1),
                       "etapa": k, "erro_max_x_limiar": round(float((g["erro"] / g["limiar"]).max()), 2),
                       "sensor_principal": contrib.idxmax(),
                       "peso_sensor_principal_pct": round(100 * float(contrib.max() / contrib.sum()), 0)})
    return pd.DataFrame(linhas)
