"""
B-6511502A v2: o modelo para o deploy, sem a temperatura e sem o deslocamento axial (out/2026), e o catálogo de
comportamentos que valida o que ele faz em cada situação.

Por quê (notebooks/b6511502a/sensores_b6511502a.ipynb): a falha de 2023 está na vibração do mancal LA do motor; a
temperatura TE-07A e o deslocamento axial ZE-50/51 não carregam a falha e geram alarme quando mudam por outro motivo
(mancal +8 °C → o modelo atual fica ~22% do tempo em alerta). Sem eles: mesma detecção, 0% de alerta.

1. Treino: grade do retreino (12 candidatos), 11 sensores, mesma janela do modelo atual (20/05/2022 a 17/04/2023),
   limiar μ+5σ, persistência 15 de 20 + regra de 1 h. Escolha: menor alerta na validação (últimos 15% da janela),
   desempate pela falha simulada (assinatura medida na falha de 2023). A falha real não entra na escolha.
2. Catálogo de comportamentos, para o escolhido E para o modelo atual (14 sensores), cada situação com o que se espera:
   normal; temperatura +8/+15 °C; vibração do mancal LA do motor +10/20/30/50%; todas as vibrações ×1,3; outro regime de
   operação; deslocamento axial +0,3 mm; sensor de vibração zerado (1 dia); sensor travado (2 dias); falha real de
   15/05/2023; falha simulada em 5 datas.

    python scripts/b6511502a_v2.py --remote --queue default
    python scripts/b6511502a_v2.py --baixar <task id>
"""
import argparse, json, os, shutil, sys, time, zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_simulado_b6511502a as ds  # noqa: E402
import subconjuntos_b6511502a as sb  # noqa: E402

TS, TE = pd.Timestamp("2022-05-20"), pd.Timestamp("2023-04-18")      # a janela do modelo atual
FEATURES = sb.VARIANTES["sem_temp_axial"]
FIM_NORMAL = ds.FALHA - pd.Timedelta(days=7)
D0 = pd.Timestamp("2023-02-01")                                       # início das mudanças permanentes
VE50B, VE51B = "VIB. MANCAL RADIAL MOT LA 90° VE-50B", "VIB. MANCAL RADIAL MOT LA 0° VE-51B"
DATAS_SINT = ["2022-11-15", "2022-12-15", "2023-01-20", "2023-03-10", "2023-04-05"]
RAMPA, PLATO = 24, 12                                                 # a falha de 2023 é súbita


def mudar(raw, inicio, fim, mudanca):
    out = raw.copy()
    m = (out.index >= inicio) & (out.index < fim) & (out[ds.CORR] > 60)
    for c, (op, v) in mudanca.items():
        if op == "+":
            out.loc[m, c] = out.loc[m, c] + v
        elif op == "*":
            out.loc[m, c] = out.loc[m, c] * v
        elif op == "=":
            out.loc[m, c] = v
        elif op == "trava":
            out.loc[m, c] = out.loc[out.index < inicio, c].dropna().iloc[-1]
    return out


FIM = pd.Timestamp("2023-06-01")
# nome → (mudança, início, fim da mudança, fim da janela avaliada, o que se espera)
CATALOGO = {
    "normal (sem mudança)": ({}, D0, FIM, FIM_NORMAL, "não alerta"),
    "mancal LNA do motor +8 °C": ({sb.TEMP: ("+", 8.0)}, D0, FIM, FIM_NORMAL, "não alerta (temperatura vai para o monitor)"),
    "mancal LNA do motor +15 °C": ({sb.TEMP: ("+", 15.0)}, D0, FIM, FIM_NORMAL, "não alerta (temperatura vai para o monitor)"),
    "vibração do motor LA +10%": ({VE50B: ("*", 1.1), VE51B: ("*", 1.1)}, D0, FIM, FIM_NORMAL, "pode alertar (sensibilidade)"),
    "vibração do motor LA +20%": ({VE50B: ("*", 1.2), VE51B: ("*", 1.2)}, D0, FIM, FIM_NORMAL, "pode alertar (sensibilidade)"),
    "vibração do motor LA +30%": ({VE50B: ("*", 1.3), VE51B: ("*", 1.3)}, D0, FIM, FIM_NORMAL, "alerta"),
    "vibração do motor LA +50%": ({VE50B: ("*", 1.5), VE51B: ("*", 1.5)}, D0, FIM, FIM_NORMAL, "alerta"),
    "todas as vibrações ×1,3": ({c: ("*", 1.3) for c in ds.VIB}, D0, FIM, FIM_NORMAL, "alerta"),
    "outro regime (descarga −8 bar, corrente −15 A)": (ds.CENARIOS["regime"], D0, FIM, FIM_NORMAL, "não alerta (normal novo: retreino)"),
    "deslocamento axial +0,3 mm": ({c: ("+", 0.3) for c in sb.AXIAL}, D0, FIM, FIM_NORMAL, "não alerta (fora do modelo: regra de limite)"),
    "sensor VE-50B zerado (1 dia)": ({VE50B: ("=", 0.0)}, pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-02"),
                                     pd.Timestamp("2023-03-03"), "alerta (falha de instrumento)"),
    "sensor VE-50B travado (2 dias)": ({VE50B: ("trava", None)}, pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-03"),
                                       pd.Timestamp("2023-03-04"), "não alerta (dado congelado: monitor M7)"),
}


def assinatura(raw):
    op = raw[raw[ds.CORR] > 60]
    return (op["2023-05-13":"2023-05-15"][FEATURES].median() - op["2022-11-01":"2023-05-05"][FEATURES].median()).to_dict()


def injetar(raw, sig, t0):
    out = raw.copy()
    dt = np.asarray((out.index - t0).total_seconds()) / 3600
    prof = np.where((dt < 0) | (dt > RAMPA + PLATO), 0, np.clip(dt / RAMPA, 0, 1)) * (out[ds.CORR] > 60).values
    for c, v in sig.items():
        out[c] = out[c].values + v * prof
    return out


def comportamento(bundle, raw, sig):
    linhas = []
    for nome, (mud, ini, fim_mud, fim_jan, espera) in CATALOGO.items():
        f = ds.alerta(bundle, mudar(raw, ini, fim_mud, mud))
        w = f[(f.index >= ini) & (f.index < fim_jan)]
        t1 = w[w].index.min() if w.any() else None
        linhas.append({"situação": nome, "esperado": espera, "% do tempo em alerta": round(100 * w.mean(), 2),
                       "episódios": ds.episodios(w),
                       "1º alerta (h depois)": round((t1 - ini).total_seconds() / 3600, 1) if t1 is not None else None})
    f = ds.alerta(bundle, raw)
    linhas.append({"situação": "falha real de 15/05/2023", "esperado": "alerta antes da falha",
                   "% do tempo em alerta": None, "episódios": None, "1º alerta (h depois)": None,
                   "antecedência (h)": ds.antecedencia_h(f)})
    for d in DATAS_SINT:
        t0 = pd.Timestamp(d); tf = t0 + pd.Timedelta(hours=RAMPA)
        g = ds.alerta(bundle, injetar(raw, sig, t0)).loc[t0:tf + pd.Timedelta(hours=PLATO)]
        t1 = g[g].index.min() if g.any() else None
        linhas.append({"situação": f"falha simulada em {t0:%d/%m/%Y}", "esperado": "alerta antes do fim da rampa (24 h)",
                       "antecedência (h)": round((tf - t1).total_seconds() / 3600, 1) if t1 is not None else None,
                       "base_ja_alertava": bool(f.loc[t0:tf + pd.Timedelta(hours=PLATO)].any()),
                       "bomba_operando_%": round(100 * (raw.loc[t0:tf + pd.Timedelta(hours=PLATO), ds.CORR] > 60).mean(), 1)})
    return pd.DataFrame(linhas)


def rodar(out: Path, remote: bool, rapido: bool):
    raw = ds.load_equipment_data(ds.EQ, from_clearml=remote)
    sig = assinatura(raw)
    preset = ds.get_preprocessing_steps(ds.EQ, "baseline")
    steps = deepcopy(ds.CFG.pre_split_steps); steps[-1]["features"] = FEATURES
    pre, _, _ = ds.run_preprocessing(raw, steps)
    tr = pre[(pre.index >= TS) & (pre.index < TE)]
    n_val = int(len(tr) * 0.15); fit, va = tr.iloc[:-n_val], tr.iloc[-n_val:]
    Xtr, art, _ = ds.run_preprocessing(fit, preset, return_artifacts=True, return_report=True)
    Xva = ds.run_preprocessing(va, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    full = ds.run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
    dl = lambda X, sh: DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float32)), batch_size=256, shuffle=sh)
    cands = []
    for layers, latent, lr in (ds.GRID[:1] if rapido else ds.GRID):
        for seed in (ds.SEEDS[:2] if rapido else ds.SEEDS):
            tag = f"l{'-'.join(map(str, layers))}_lat{latent}_lr{lr:g}_s{seed}"
            t0 = time.time()
            m = ds.build_model("vae", Xtr.shape[1], dense_layers=list(layers), latent_dim=latent)
            torch.manual_seed(seed); np.random.seed(seed)
            m = ds.train_vae(m, dl(Xtr, True), dl(Xva, False), epochs=2 if rapido else 60, learning_rate=lr,
                             weight_decay=1e-5, patience=10)
            m.eval(); torch.manual_seed(0)
            e = pd.Series(ds.compute_vae_errors(m, full), index=full.index)
            et = e[(e.index >= TS) & (e.index < TE)]
            b = sb.salvar(out / "cands" / tag, m, latent, layers, art, float(et.mean()), float(et.std()), steps, preset)
            f = ds.alerta(b, raw)
            sint = []
            for d in DATAS_SINT:
                t = pd.Timestamp(d); tf = t + pd.Timedelta(hours=RAMPA)
                g = ds.alerta(b, injetar(raw, sig, t)).loc[t:tf + pd.Timedelta(hours=PLATO)]
                sint.append((tf - g[g].index.min()).total_seconds() / 3600 if g.any() else -99.0)
            cands.append({"cand": tag, "fp_val_%": 100 * f[(f.index >= va.index.min()) & (f.index < TE)].mean(),
                          "alerta_treino_%": 100 * f[(f.index >= TS) & (f.index < TE)].mean(),
                          "alerta_depois_treino_%": 100 * f[(f.index >= TE) & (f.index < FIM_NORMAL)].mean(),
                          "sintetica_mediana_h": float(np.median(sint)), "sintetica_detecta": int(sum(s > 0 for s in sint)),
                          "falha_real_h": ds.antecedencia_h(f)})
            print(cands[-1], f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(cands).to_csv(out / "candidatos.csv", index=False)
    c = pd.DataFrame(cands)
    esc = c.sort_values(["fp_val_%", "sintetica_detecta", "sintetica_mediana_h"], ascending=[True, False, False]).iloc[0]
    print("ESCOLHIDO", esc.to_dict(), flush=True)
    dest = out / f"model_{TS.date()}_{TE.date()}_VAE_11s"
    shutil.copytree(out / "cands" / esc["cand"], dest, dirs_exist_ok=True)
    alarm = json.loads((dest / "alarm.json").read_text()); alarm["threshold_calibration"]["candidate"] = esc["cand"]
    (dest / "alarm.json").write_text(json.dumps(alarm, indent=1, ensure_ascii=False))
    shutil.rmtree(out / "cands")
    tabs = []
    for nome, b in [("novo (11 sensores)", dest), ("atual (14 sensores)", ds.PRODUCAO)]:
        t = comportamento(b, raw, sig); t.insert(0, "modelo", nome); tabs.append(t)
        print(t.to_string(), flush=True)
    pd.concat(tabs).to_csv(out / "catalogo.csv", index=False)
    pd.Series(sig).to_csv(out / "assinatura_falha_2023.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/b6511502a_v2")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--remote", action="store_true"); ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None); ap.add_argument("--clearml-task-name", default="b6511502a-v2")
    a = ap.parse_args()
    if a.baixar:
        from clearml import Task
        out = ds.ROOT / a.out
        z = Task.get_task(task_id=a.baixar).artifacts["resultado"].get_local_copy(extract_archive=False)
        out.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(z).extractall(out)
        print(f"resultado da task {a.baixar} extraído em {out}"); return 0
    task = None
    if a.remote or os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        Task.add_requirements("pyarrow"); Task.add_requirements("torch", package_version="")
        task = Task.init(project_name="Transpetro", task_name=a.clearml_task_name, output_uri=True, reuse_last_task_id=False)
        task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
        task.connect(vars(a)); a.remote = True
        task.execute_remotely(queue_name=a.queue)
    out = ds.ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    try:
        rodar(out, a.remote, a.rapido)
    finally:
        if task is not None:
            if (out / "catalogo.csv").exists():
                task.upload_artifact("catalogo", artifact_object=out / "catalogo.csv")
            task.upload_artifact("resultado", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
