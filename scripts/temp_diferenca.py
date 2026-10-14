"""
Experimento: temperatura do mancal em NÍVEL × em DIFERENÇA (mancal da bomba − mancal do motor) no modelo de anomalia do
B-8802B. A temperatura do mancal LA muda ~6 °C com a estação, o mesmo tamanho das mudanças reais (28/03/2026: −6 °C), e
em 2026 caiu abaixo do piso do clip. O mancal do motor sente o mesmo ambiente: a diferença tira a estação.

Variantes (config.py), todas com o mesmo dado, filtros, janela (treino 2025, held-out jan–mai/2026), grade e bateria:
  nivel  B-8802B-2025      entrada atual (Temperatura Bomba LA)
  dT1    B-8802B-2025-dT1  Temperatura Bomba LA → Δ bomba LA − motor LA
  dT2    B-8802B-2025-dT2  dT1 + Δ bomba LNA − motor LNA

Etapas: (1) scripts/retrain_pipeline.py por variante (grade 4 cfg × 3 sementes + bateria dos 3 melhores);
(2) avaliação de TODOS os 12 candidatos de cada variante com a régua de deploy (μ+6,5σ, 15 de 20 no tempo, alerta ≥ 1 h):
alerta em 2025 / jan–mai/26 / jun–ago/26, episódios em 2026, saturação do clip da temperatura em abr–ago/26, falha real
de 2022 e falha sintética em 5 datas de 2026 com a assinatura de 2022 nos 8 sensores (inclui o motor esquentando, para a
diferença não levar vantagem injusta).

    python scripts/temp_diferenca.py --remote --queue default    # worker do ClearML
    python scripts/temp_diferenca.py --baixar <task id>          # traz o resultado para --out
"""
import argparse, json, os, pickle, shutil, subprocess, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_pipeline import _baixar_csvs  # noqa: E402

VAR = {"nivel": "B-8802B-2025", "dT1": "B-8802B-2025-dT1", "dT2": "B-8802B-2025-dT2"}
TS, TE, HE = "2025-01-01", "2026-01-01", "2026-06-01"
DATAS = ["2026-01-25", "2026-03-05", "2026-04-20", "2026-06-10", "2026-07-15"]
RAMP, HOLD = 48, 24


def treinar(out: Path, remote: bool):
    procs = []
    for nome, eq in VAR.items():
        cmd = [sys.executable, str(ROOT / "scripts/retrain_pipeline.py"), "--equipment", eq, "--train-start", TS,
               "--train-end", TE, "--heldout-end", HE, "--out", str(out / nome), "--operacao-confirmou"]
        cmd += ["--from-clearml"] if remote else []
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        log = open(out / f"{nome}.log", "w")
        procs.append((nome, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)))
    for nome, p in procs:
        print(f"[treino] {nome}: código {p.wait()} (1 = nenhum candidato aprovado na bateria, não é erro)", flush=True)


def avaliar(out: Path, remote: bool, logger=None) -> pd.DataFrame:
    import torch
    from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps
    from transpetro_modelos.data.loading import load_equipment_data
    from transpetro_modelos.data.preprocessing import run_preprocessing
    from transpetro_modelos.training.automl import build_model
    from transpetro_modelos.training.evaluate import compute_vae_errors
    from transpetro_modelos.drift import battery
    si = battery.si

    ts, te, he = map(pd.Timestamp, (TS, TE, HE))
    # assinatura medida na rampa de 2022 nos 8 sensores (os 3 extras: experimento FE de ago/2026)
    sig = dict(battery.BATTERY["B-8802B-2025"]["synthetic"]["signature"],
               **{"Temperatura Bomba LNA": -1.7, "Temperatura Motor LA": 2.6, "Temperatura Motor LNA": 1.8})
    raw = si.carregar_dados(battery.BATTERY["B-8802B-2025"]["data_new"])
    ce = battery.BATTERY["B-8802B-2025"]["cross_era"]
    raw22 = si.carregar_dados(next(Path(ce["csv"]).rglob(ce["csv_glob"])))
    fail, r0 = pd.Timestamp(ce["failure"]), pd.Timestamp(ce["ramp_start"])

    def episodios(f):
        x = f[f]
        return 0 if x.empty else int((x.index.to_series().diff() > pd.Timedelta("1h")).sum() + 1)

    linhas = []
    for var, eq in VAR.items():
        cfg = EQUIPMENT_CONFIGS[eq]; preset = get_preprocessing_steps(eq, "baseline")
        pre, _, _ = run_preprocessing(load_equipment_data(eq, from_clearml=remote), cfg.pre_split_steps)
        tr = pre[(pre.index >= ts) & (pre.index < te)]; n_val = int(len(tr) * 0.15)   # mesmo corte do retrain.py
        Xtr, art, _ = run_preprocessing(tr.iloc[:-n_val], preset, return_artifacts=True, return_report=True)
        full = run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
        tcol = [c for c in Xtr.columns if "Temp" in c]
        lo, hi = (np.array(v) for v in zip(*[art.clip_bounds[c] for c in tcol]))
        z = pre.loc["2026-04-01":, tcol]
        sat = 100 * ((z < lo) | (z > hi)).any(axis=1).mean()
        for pt in sorted((out / var / "cands").glob("*.pt")):
            tag = pt.stem; lay = [int(v) for v in tag.split("_")[0][1:].split("-")]
            m = build_model("vae", Xtr.shape[1], dense_layers=lay, latent_dim=16)
            m.load_state_dict(torch.load(pt, map_location="cpu")); m.eval()
            torch.manual_seed(0); e = pd.Series(compute_vae_errors(m, full), index=full.index)
            mtr = e[(e.index >= ts) & (e.index < te)]; mu, sd = float(mtr.mean()), float(mtr.std())
            b = out / "bundles" / var / tag; b.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), b / "model_state.pt")
            (b / "model_arch.json").write_text(json.dumps({"model_type": "vae", "input_dim": Xtr.shape[1],
                                                           "encoding_layers": lay, "latent_dim": 16}))
            pickle.dump(art.scaler, open(b / "scaler.pkl", "wb"))
            (b / "clip_bounds.json").write_text(json.dumps({c: list(v) for c, v in art.clip_bounds.items()}, ensure_ascii=False))
            (b / "pipeline.json").write_text(json.dumps(list(cfg.pre_split_steps) + preset, ensure_ascii=False))
            (b / "alarm.json").write_text(json.dumps({
                "model_type": "vae", "threshold": mu + 6.5 * sd, "threshold_attention": mu + 4 * sd,
                "debounce_window": 20, "debounce_min": 15, "persistence_mode": "time", "persistence_step": "5min",
                "min_alert_hours": 1.0, "features": list(Xtr.columns)}, ensure_ascii=False))
            torch.manual_seed(0); f = battery.score(b, raw)["is_anomaly"]
            torch.manual_seed(0); f22 = battery.score(b, raw22)["is_anomaly"]
            rp = f22[(f22.index >= r0) & (f22.index < fail)]; first = rp[rp].index.min() if rp.any() else None
            lin = {"variante": var, "cand": tag,
                   "alerta_2025_%": 100 * f[f.index < te].mean(),
                   "alerta_jan-mai26_%": 100 * f[(f.index >= te) & (f.index < he)].mean(),
                   "alerta_jun-ago26_%": 100 * f[f.index >= he].mean(),
                   "episodios_2026": episodios(f[f.index >= te]),
                   "clip_temp_abr-ago26_%": sat,
                   "falha2022_dias": (fail - first).total_seconds() / 86400 if first is not None else np.nan,
                   "normal2022_%": 100 * f22[f22.index < r0].mean()}
            for d in DATAS:
                t0 = pd.Timestamp(d); tf = t0 + pd.Timedelta(hours=RAMP)
                if f[(f.index >= t0) & (f.index <= tf + pd.Timedelta(hours=HOLD))].any():
                    lin[f"sint_{d[5:]}_h"] = np.nan; lin[f"sint_{d[5:]}_contaminada"] = True; continue
                jan = raw.loc[t0 - pd.Timedelta(days=3): tf + pd.Timedelta(hours=HOLD + 24)]
                torch.manual_seed(0)
                w = battery.score(b, battery.inject(jan, sig, t0, RAMP, HOLD, 1.0))["is_anomaly"]
                w = w.loc[t0: tf + pd.Timedelta(hours=HOLD)]
                fi = w[w].index.min() if w.any() else None
                lin[f"sint_{d[5:]}_h"] = round((tf - fi).total_seconds() / 3600, 1) if fi is not None else np.nan
            linhas.append(lin)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lin.items()}, flush=True)
            pd.DataFrame(linhas).to_csv(out / "avaliacao.csv", index=False)
    tab = pd.DataFrame(linhas)
    if logger is not None:
        logger.report_table("avaliacao", "candidatos", iteration=0, table_plot=tab)
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/temp_diferenca")
    ap.add_argument("--remote", action="store_true", help="enfileira no ClearML e sai")
    ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None, help="id da task do ClearML: baixa o artifact `temp_diferenca` para --out e sai")
    ap.add_argument("--so-avaliar", action="store_true", help="pula o treino (usa os candidatos em --out)")
    ap.add_argument("--clearml-task-name", default="temp-diferenca-b8802b")
    a = ap.parse_args()
    if a.baixar:
        from clearml import Task
        out = ROOT / a.out
        z = Task.get_task(task_id=a.baixar).artifacts["temp_diferenca"].get_local_copy(extract_archive=False)
        out.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(z).extractall(out)
        print(f"resultado da task {a.baixar} extraído em {out}"); return 0

    task = logger = None
    if a.remote or os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        Task.add_requirements("pyarrow"); Task.add_requirements("torch", package_version="")
        task = Task.init(project_name="Transpetro", task_name=a.clearml_task_name, output_uri=True, reuse_last_task_id=False)
        task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
        task.connect(vars(a))
        a.remote = True
        task.execute_remotely(queue_name=a.queue)       # daqui para baixo, só no worker
        logger = task.get_logger()
        _baixar_csvs()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if not a.so_avaliar:
        treinar(out, a.remote)
    tab = avaliar(out, a.remote, logger)
    print(tab.groupby("variante").median(numeric_only=True).round(3).to_string())
    if task is not None:
        task.upload_artifact("avaliacao", artifact_object=out / "avaliacao.csv")
        task.upload_artifact("temp_diferenca", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
