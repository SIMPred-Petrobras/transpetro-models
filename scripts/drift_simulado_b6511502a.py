"""
B-6511502A: teste do ciclo de retreino com uma mudança de conceito SIMULADA (out/2026).

Não há dado do B-6511502A depois de 05/2023, então não dá para saber se o normal dele mudou. Este experimento prepara
e testa a pipeline do equipamento: a partir de 01/11/2022, o dado real recebe uma mudança permanente (só com a bomba
operando), como as que vimos no B-8802B depois do reparo, e o ciclo roda como se fosse produção:

  cenários   temperatura  mancal LNA do motor (TE-07A) +8 °C
             vibracao     as 8 vibrações radiais ×1,3
             regime       pressão de descarga −8 bar e corrente do motor −15 A (outro ponto de operação)
             sem_mudanca  controle: o dado real, sem mudança
  por cenário
    - o modelo em produção (bundle deploy_v2) sobre o dado alterado: alerta e falha real de 15/05/2023;
    - o monitor (M6, KS diário) com a referência do treino do modelo: em quantos dias percebe a mudança;
    - o ciclo: modelos com 1, 3 e 6 meses de dado depois da mudança (12 meses não cabem: o dado acaba em 05/2023),
      grade do retreino (12 candidatos), limiar μ+5σ do equipamento, persistência 15 de 20 + regra de 1 h.
      Escolha do candidato: menor alerta na validação (últimos 15% da janela). A falha real NÃO entra na escolha:
      é o teste.

    python scripts/drift_simulado_b6511502a.py --remote --queue default     # worker do ClearML
    python scripts/drift_simulado_b6511502a.py --baixar <task id>           # traz o resultado para --out
"""
import argparse, json, os, pickle, shutil, sys, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transpetro_modelos.config import EQUIPMENT_CONFIGS, get_preprocessing_steps  # noqa: E402
from transpetro_modelos.data.loading import load_equipment_data  # noqa: E402
from transpetro_modelos.data.preprocessing import run_preprocessing  # noqa: E402
from transpetro_modelos.training.automl import build_model  # noqa: E402
from transpetro_modelos.training.train import train_vae  # noqa: E402
from transpetro_modelos.training.evaluate import compute_vae_errors  # noqa: E402
from transpetro_modelos.drift import battery, monitor as mon  # noqa: E402
from transpetro_modelos.drift.detectors import CalibratedKSDetector  # noqa: E402
from transpetro_modelos.drift.retrain import GRID, SEEDS  # noqa: E402

EQ = "B-6511502A"
CFG = EQUIPMENT_CONFIGS[EQ]
PRODUCAO = ROOT / "deploy_v2/Transpetro/B-6511502A/modelos/model_2022-05-17_2023-05-17_VAE"
MUDANCA = pd.Timestamp("2022-11-01")
FALHA = pd.Timestamp(CFG.failure_date)
FIM_NORMAL = FALHA - pd.Timedelta(days=7)          # depois disso o dado já é a falha (o sinal aparece no último dia)
Y_ALARM = json.loads((PRODUCAO / "alarm.json").read_text())["threshold_calibration"]["y_alarm"]   # 5,0 neste equipamento
CORR = "CORRENTE ELÉTRICA DO MOTOR"
VIB = [c for c in CFG.pre_split_steps[-1]["features"] if c.startswith("VIB.")]
CENARIOS = {
    "sem_mudanca": {},
    "temperatura": {"TEMP. MANCAL LNA MOT TE-07A1/A2": ("+", 8.0)},
    "vibracao": {c: ("*", 1.3) for c in VIB},
    "regime": {"PRESSÃO DESCARGA": ("+", -8.0), CORR: ("+", -15.0)},
}
ETAPAS = [("1 mês", 1), ("3 meses", 3), ("6 meses", 6)]


def aplicar(raw: pd.DataFrame, mudanca: dict) -> pd.DataFrame:
    out = raw.copy()
    m = (out.index >= MUDANCA) & (out[CORR] > 60)
    for c, (op, v) in mudanca.items():
        out.loc[m, c] = out.loc[m, c] + v if op == "+" else out.loc[m, c] * v
    return out


def salvar_bundle(pasta: Path, m, latent, layers, art, mu, sd, ts, te, preset) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), pasta / "model_state.pt")
    (pasta / "model_arch.json").write_text(json.dumps({"model_type": "vae", "input_dim": len(art.clip_bounds),
                                                       "encoding_layers": list(layers), "latent_dim": latent}))
    pickle.dump(art.scaler, open(pasta / "scaler.pkl", "wb"))
    (pasta / "clip_bounds.json").write_text(json.dumps({c: list(v) for c, v in art.clip_bounds.items()}, ensure_ascii=False))
    (pasta / "pipeline.json").write_text(json.dumps(list(CFG.pre_split_steps) + preset, ensure_ascii=False))
    (pasta / "alarm.json").write_text(json.dumps({
        "model_type": "vae", "threshold": mu + Y_ALARM * sd, "threshold_attention": mu + 4 * sd,
        "debounce_window": 20, "debounce_min": 15, "persistence_mode": "time", "persistence_step": "5min",
        "min_alert_hours": 1.0, "features": list(art.clip_bounds),
        "threshold_calibration": {"method": "sigma", "mean_normal": mu, "std_normal": sd, "y_alarm": Y_ALARM,
                                  "normal_window": {"start": str(ts), "end": str(te)}}}, ensure_ascii=False))
    return pasta


def alerta(bundle: Path, raw: pd.DataFrame) -> pd.Series:
    torch.manual_seed(0)
    return battery.score(bundle, raw)["is_anomaly"].astype(bool)


def episodios(f: pd.Series) -> int:
    x = f[f]
    return 0 if x.empty else int((x.index.to_series().diff() > pd.Timedelta("1h")).sum() + 1)


def antecedencia_h(f: pd.Series):
    w = f[(f.index >= FALHA - pd.Timedelta(days=5)) & (f.index < FALHA)]
    return round((FALHA - w[w].index.min()).total_seconds() / 3600, 1) if w.any() else None


def m6(bundle: Path, raw: pd.DataFrame, ref_ini, ref_fim, desde, ate) -> list:
    """Dias com disparo do monitor (KS diário) entre `desde` e `ate`, com a referência no treino do modelo."""
    df = mon._sem_congelado(bundle, mon._temporal_steps(bundle, raw))
    ref = df[(df.index >= pd.Timestamp(ref_ini)) & (df.index <= pd.Timestamp(ref_fim))]
    det = CalibratedKSDetector(ref, window_size=mon.M6_WIN, k_consecutive=mon.M6_KCONSEC,
                               persistence_window=mon.M6_NWIN, n_reference=mon.M6_NREF, sampling="quantile")
    alvo = df[(df.index > pd.Timestamp(ref_fim)) & (df.index >= desde) & (df.index < ate)]
    return [str(t.date()) for t, _ in det.scan(alvo, reset_gap=mon.M6_RESET_GAP)[1]]


def rodar(out: Path, remote: bool, cenarios: list[str], rapido: bool = False):
    raw0 = load_equipment_data(EQ, from_clearml=remote)
    preset = get_preprocessing_steps(EQ, "baseline")
    dl = lambda X, sh: DataLoader(TensorDataset(torch.tensor(X.values, dtype=torch.float32)), batch_size=256, shuffle=sh)
    prod_nw = json.loads((PRODUCAO / "alarm.json").read_text())["threshold_calibration"]["normal_window"]
    cands, etapas = [], []
    for cen in cenarios:
        raw = aplicar(raw0, CENARIOS[cen])
        (out / cen).mkdir(parents=True, exist_ok=True)
        # o modelo em produção diante da mudança
        f = alerta(PRODUCAO, raw)
        f.to_frame("producao").to_parquet(out / cen / "alerta_producao.parquet")
        fires = m6(PRODUCAO, raw, prod_nw["start"], MUDANCA - pd.Timedelta(minutes=5), MUDANCA, FIM_NORMAL)
        etapas.append({"cenario": cen, "etapa": "produção (sem retreino)", "treino": f"{prod_nw['start'][:10]} → {prod_nw['end'][:10]}",
                       "alerta_pos_mudanca_%": 100 * f[(f.index >= MUDANCA) & (f.index < FIM_NORMAL)].mean(),
                       "episodios_pos_mudanca": episodios(f[(f.index >= MUDANCA) & (f.index < FIM_NORMAL)]),
                       "falha_antecedencia_h": antecedencia_h(f), "m6_primeiro_disparo": fires[0] if fires else None,
                       "m6_dias": len(fires)})
        print(etapas[-1], flush=True)
        if cen == "sem_mudanca":
            continue
        pre, _, _ = run_preprocessing(raw, CFG.pre_split_steps)
        for nome, meses in (ETAPAS[:1] if rapido else ETAPAS):
            ts, te = MUDANCA, MUDANCA + pd.DateOffset(months=meses)
            op_fim = MUDANCA + pd.DateOffset(months={1: 3, 3: 6, 6: 12}[meses])     # entra em te e roda até o próximo
            tr = pre[(pre.index >= ts) & (pre.index < te)]
            n_val = int(len(tr) * 0.15); fit, va = tr.iloc[:-n_val], tr.iloc[-n_val:]
            Xtr, art, _ = run_preprocessing(fit, preset, return_artifacts=True, return_report=True)
            Xva = run_preprocessing(va, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
            full = run_preprocessing(pre, preset, fitted_artifacts=art, return_artifacts=True, return_report=True)[0]
            linhas = []
            for layers, latent, lr in (GRID[:1] if rapido else GRID):
                for seed in (SEEDS[:1] if rapido else SEEDS):
                    tag = f"l{'-'.join(map(str, layers))}_lat{latent}_lr{lr:g}_s{seed}"
                    t0 = time.time()
                    m = build_model("vae", Xtr.shape[1], dense_layers=list(layers), latent_dim=latent)
                    torch.manual_seed(seed); np.random.seed(seed)
                    m = train_vae(m, dl(Xtr, True), dl(Xva, False), epochs=2 if rapido else 60, learning_rate=lr, weight_decay=1e-5, patience=10)
                    m.eval(); torch.manual_seed(0)
                    e = pd.Series(compute_vae_errors(m, full), index=full.index)
                    et = e[(e.index >= ts) & (e.index < te)]
                    b = salvar_bundle(out / cen / f"m{meses:02d}" / tag, m, latent, layers, art,
                                      float(et.mean()), float(et.std()), ts, te, preset)
                    fa = alerta(b, raw)
                    lin = {"cenario": cen, "etapa": nome, "cand": tag,
                           "fp_val_%": 100 * fa[(fa.index >= va.index.min()) & (fa.index < te)].mean(),
                           "alerta_operacao_%": 100 * fa[(fa.index >= te) & (fa.index < min(op_fim, FIM_NORMAL))].mean()
                           if te < FIM_NORMAL else np.nan,
                           "episodios_operacao": episodios(fa[(fa.index >= te) & (fa.index < min(op_fim, FIM_NORMAL))])
                           if te < FIM_NORMAL else 0,
                           "falha_antecedencia_h": antecedencia_h(fa)}
                    linhas.append((lin, b, fa))
                    cands.append(lin)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lin.items()}, f"{time.time() - t0:.0f}s", flush=True)
            ordem = {lin["cand"]: i for i, (lin, _, _) in enumerate(linhas)}
            lin, b, fa = min(linhas, key=lambda x: (round(x[0]["fp_val_%"], 4), ordem[x[0]["cand"]]))
            fa.to_frame(nome).to_parquet(out / cen / f"alerta_m{meses:02d}.parquet")
            fires = m6(b, raw, ts, te, te, min(op_fim, FIM_NORMAL)) if te < FIM_NORMAL else []
            etapas.append({"cenario": cen, "etapa": nome, "treino": f"{ts.date()} → {te.date()}", "escolhido": lin["cand"],
                           "fp_val_%": lin["fp_val_%"], "alerta_pos_mudanca_%": lin["alerta_operacao_%"],
                           "episodios_pos_mudanca": lin["episodios_operacao"],
                           "falha_antecedencia_h": lin["falha_antecedencia_h"],
                           "candidatos_que_detectam_falha": sum(x[0]["falha_antecedencia_h"] is not None for x in linhas),
                           "m6_primeiro_disparo": fires[0] if fires else None, "m6_dias": len(fires)})
            print("ETAPA", etapas[-1], flush=True)
            for _, bb, _ in linhas:                     # só o escolhido fica (os bundles pesam pouco, mas são 12 por etapa)
                if bb != b:
                    shutil.rmtree(bb)
            pd.DataFrame(cands).to_csv(out / "candidatos.csv", index=False)
            pd.DataFrame(etapas).to_csv(out / "etapas.csv", index=False)
    pd.DataFrame(etapas).to_csv(out / "etapas.csv", index=False)
    return pd.DataFrame(etapas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/drift_simulado_b6511502a")
    ap.add_argument("--cenarios", nargs="*", default=list(CENARIOS))
    ap.add_argument("--rapido", action="store_true", help="teste do código: 1 etapa, 1 candidato, 2 épocas")
    ap.add_argument("--remote", action="store_true"); ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None); ap.add_argument("--clearml-task-name", default="drift-simulado-b6511502a")
    a = ap.parse_args()
    if a.baixar:
        from clearml import Task
        out = ROOT / a.out
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
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    try:
        tab = rodar(out, a.remote, a.cenarios, a.rapido)
        print(tab.round(3).to_string())
    finally:
        if task is not None:
            if (out / "etapas.csv").exists():
                task.upload_artifact("etapas", artifact_object=out / "etapas.csv")
            task.upload_artifact("resultado", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
