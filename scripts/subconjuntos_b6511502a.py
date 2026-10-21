"""
B-6511502A: quais sensores o modelo de anomalia precisa? (out/2026)

A falha de 15/05/2023 aparece na vibração do mancal LA do motor (VE-50B/VE-51B, +5 a +8 desvios no último dia); as
temperaturas de mancal quase não mudam (TE-07A, a que o modelo usa: +0,3 desvio). E só o mancal 8 °C mais quente basta
para o modelo atual ficar 15% do tempo em alerta (scripts/drift_simulado_b6511502a.py). Variantes de entrada:

  atual          os 14 sensores do modelo em produção
  sem_temp       sem a temperatura TE-07A (13)
  sem_temp_axial sem TE-07A e sem os deslocamentos axiais ZE-50/51 (11)
  so_vib         só as 8 vibrações radiais

Cada variante: grade do retreino (12 candidatos), treino de 20/05/2022 a 31/01/2023, limiar μ+5σ (o do equipamento),
persistência 15 de 20 + regra de 1 h. Teste em fev–mai/2023 (nenhum modelo viu): alerta no normal (até 08/05), a falha
real (antecedência) e, a partir de 01/02/2023, as mudanças simuladas "temperatura +8 °C" e "vibrações ×1,3" — o modelo
não deve alertar por uma temperatura mais alta; numa vibração 30% maior, alertar é esperado.

    python scripts/subconjuntos_b6511502a.py --remote --queue default
    python scripts/subconjuntos_b6511502a.py --baixar <task id>
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

TS, TE = pd.Timestamp("2022-05-20"), pd.Timestamp("2023-02-01")
FIM_NORMAL = ds.FALHA - pd.Timedelta(days=7)
TEMP, AXIAL = "TEMP. MANCAL LNA MOT TE-07A1/A2", ["DESLOC. AXIAL EIXO BB LNA 1 ZE-50", "DESLOC. AXIAL EIXO BB LNA 2 ZE-51"]
TODOS = ds.CFG.pre_split_steps[-1]["features"]
VARIANTES = {"atual": TODOS,
             "sem_temp": [c for c in TODOS if c != TEMP],
             "sem_temp_axial": [c for c in TODOS if c != TEMP and c not in AXIAL],
             "so_vib": ds.VIB}
CENARIOS = {"real": {}, "temperatura": ds.CENARIOS["temperatura"], "vibracao": ds.CENARIOS["vibracao"]}


def aplicar(raw, mudanca):
    out = raw.copy()
    m = (out.index >= TE) & (out[ds.CORR] > 60)
    for c, (op, v) in mudanca.items():
        out.loc[m, c] = out.loc[m, c] + v if op == "+" else out.loc[m, c] * v
    return out


def rodar(out: Path, remote: bool, variantes: list[str], rapido: bool):
    raw0 = ds.load_equipment_data(ds.EQ, from_clearml=remote)
    raws = {k: aplicar(raw0, v) for k, v in CENARIOS.items()}
    preset = ds.get_preprocessing_steps(ds.EQ, "baseline")
    dl = lambda X, sh: DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float32)), batch_size=256, shuffle=sh)
    linhas = []
    for var in variantes:
        steps = deepcopy(ds.CFG.pre_split_steps); steps[-1]["features"] = VARIANTES[var]
        pre, _, _ = ds.run_preprocessing(raw0, steps)
        tr = pre[(pre.index >= TS) & (pre.index < TE)]
        n_val = int(len(tr) * 0.15); fit, va = tr.iloc[:-n_val], tr.iloc[-n_val:]
        Xtr, art, _ = ds.run_preprocessing(fit, preset, return_artifacts=True, return_report=True)
        Xva = ds.run_preprocessing(va, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
        full = ds.run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
        for layers, latent, lr in (ds.GRID[:1] if rapido else ds.GRID):
            for seed in (ds.SEEDS[:1] if rapido else ds.SEEDS):
                tag = f"l{'-'.join(map(str, layers))}_lat{latent}_lr{lr:g}_s{seed}"
                t0 = time.time()
                m = ds.build_model("vae", Xtr.shape[1], dense_layers=list(layers), latent_dim=latent)
                torch.manual_seed(seed); np.random.seed(seed)
                m = ds.train_vae(m, dl(Xtr, True), dl(Xva, False), epochs=2 if rapido else 60, learning_rate=lr,
                                 weight_decay=1e-5, patience=10)
                m.eval(); torch.manual_seed(0)
                e = pd.Series(ds.compute_vae_errors(m, full), index=full.index)
                et = e[(e.index >= TS) & (e.index < TE)]
                pasta = out / var / tag
                b = salvar(pasta, m, latent, layers, art, float(et.mean()), float(et.std()), steps, preset)
                lin = {"variante": var, "n_sensores": len(VARIANTES[var]), "cand": tag}
                for cen, raw in raws.items():
                    f = ds.alerta(b, raw)
                    teste = f[(f.index >= TE) & (f.index < FIM_NORMAL)]
                    lin[f"{cen}_alerta_%"] = 100 * teste.mean()
                    lin[f"{cen}_episodios"] = ds.episodios(teste)
                    lin[f"{cen}_falha_h"] = ds.antecedencia_h(f)
                    if cen == "real":
                        lin["treino_alerta_%"] = 100 * f[(f.index >= TS) & (f.index < TE)].mean()
                        f.to_frame(tag).to_parquet(pasta / "alerta_real.parquet")
                linhas.append(lin)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lin.items()}, f"{time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(linhas).to_csv(out / "avaliacao.csv", index=False)
    return pd.DataFrame(linhas)


def salvar(pasta, m, latent, layers, art, mu, sd, steps, preset, ts=TS, te=TE):
    b = ds.salvar_bundle(pasta, m, latent, layers, art, mu, sd, ts, te, preset)
    (b / "pipeline.json").write_text(json.dumps(list(steps) + preset, ensure_ascii=False))   # com o select_features da variante
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/subconjuntos_b6511502a")
    ap.add_argument("--variantes", nargs="*", default=list(VARIANTES))
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--remote", action="store_true"); ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None); ap.add_argument("--clearml-task-name", default="subconjuntos-b6511502a")
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
        tab = rodar(out, a.remote, a.variantes, a.rapido)
        print(tab.groupby("variante").median(numeric_only=True).round(3).to_string())
    finally:
        if task is not None:
            if (out / "avaliacao.csv").exists():
                task.upload_artifact("avaliacao", artifact_object=out / "avaliacao.csv")
            task.upload_artifact("resultado", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
