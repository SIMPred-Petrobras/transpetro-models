"""
Teste da janela deslizante contra o retreino por confirmação, na mudança de 28/03/2026 do B-8802B (mancal LA da bomba
~6 °C mais frio depois da parada de 27/03). Treina os modelos de cada abordagem com `scripts/retrain_pipeline.py
--provisorio` (mesma grade, seleção e bateria) e sobe tudo como artifact `janela`; a comparação de alertas é feita
depois, localmente, pontuando os bundles sobre o dado.

Abordagens (todas treinadas aqui):
  confirmacao  política atual: só dado depois da mudança, modelo novo com 1 e 3 meses (28/04 e 28/06/2026)
  misto        nas mesmas datas, os últimos 12 meses (1 mês novo + 11 do normal velho, e depois 3 + 9)
  deslizante   todo mês (06/02 a 06/08/2026), os últimos 12 meses, sem esperar confirmação
  deslizante_rampa  igual à deslizante, mas com uma degradação lenta simulada na vibração LNA (+0,15 mm/s por mês
               desde 06/01/2026): mede se a janela deslizante aprende o defeito como normal (o modelo fixo de 12
               meses, comparado com ela, não precisa de treino)

    python scripts/janela_deslizante.py --remote --queue default      # worker do ClearML
    python scripts/janela_deslizante.py --baixar <task id>            # traz o resultado para --out
"""
import argparse, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_pipeline import _baixar_csvs  # noqa: E402

MUDANCA = pd.Timestamp("2026-03-28")
RAMPA = "Vibração Bomba LNA:2026-01-06:0.15"


def jobs():
    j = []
    for k in (1, 3):
        te = MUDANCA + pd.DateOffset(months=k)
        j.append({"nome": f"confirmacao_m{k:02d}", "abordagem": "confirmacao", "ts": MUDANCA, "te": te, "rampa": None})
        j.append({"nome": f"misto_m{k:02d}", "abordagem": "misto", "ts": te - pd.DateOffset(months=12), "te": te, "rampa": None})
    for te in pd.date_range("2026-02-06", "2026-08-06", freq=pd.DateOffset(months=1)):
        for ab, rampa in (("deslizante", None), ("deslizante_rampa", RAMPA)):
            j.append({"nome": f"{ab}_{te:%Y%m%d}", "abordagem": ab, "ts": te - pd.DateOffset(months=12), "te": te, "rampa": rampa})
    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipment", default="B-8802B-2025")
    ap.add_argument("--out", default="results/janela_deslizante")
    ap.add_argument("--remote", action="store_true", help="enfileira no ClearML e sai")
    ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None, help="id da task do ClearML: baixa o artifact `janela` para --out e sai")
    ap.add_argument("--so", default=None, help="roda só os jobs cujo nome contém este texto (teste local)")
    ap.add_argument("--clearml-task-name", default="janela-deslizante-b8802b")
    a = ap.parse_args()
    if a.baixar:
        from clearml import Task
        out = ROOT / a.out
        z = Task.get_task(task_id=a.baixar).artifacts["janela"].get_local_copy(extract_archive=False)
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
    linhas = []
    for i, j in enumerate(jobs()):
        if a.so and a.so not in j["nome"]:
            continue
        pasta = out / j["nome"]; res_path = pasta / "resultado.json"
        if not res_path.exists():
            pasta.mkdir(exist_ok=True)
            cmd = [sys.executable, str(ROOT / "scripts/retrain_pipeline.py"), "--equipment", a.equipment, "--provisorio",
                   "--train-start", str(j["ts"].date()), "--train-end", str(j["te"].date()), "--out", str(pasta),
                   "--operacao-confirmou"] + (["--from-clearml"] if a.remote else []) + (["--rampa", j["rampa"]] if j["rampa"] else [])
            print(f"\n=== {j['nome']}: {j['ts'].date()} → {j['te'].date()} ===", flush=True)
            with open(pasta / "log.txt", "w") as log:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            bat = sorted(pasta.glob("battery_cand*.json"))
            ultimo = json.loads(bat[-1].read_text()) if bat else {}
            bundle = next(pasta.glob("model_*_VAE_provisorio"), None)
            res_path.write_text(json.dumps({
                "nome": j["nome"], "abordagem": j["abordagem"], "treino_ini": str(j["ts"].date()),
                "treino_fim": str(j["te"].date()), "rampa": j["rampa"], "aprovado": rc == 0,
                "bundle": str(bundle.relative_to(ROOT)) if bundle else None,
                "fp_val_pct": ultimo.get("fp_val_pct"),
                "falha_2022_dias": (ultimo.get("cross_era") or {}).get("lead_days"),
                "sintetica_h": (ultimo.get("synthetic") or {}).get("100"),
            }, indent=1, ensure_ascii=False, default=str))
        r = json.loads(res_path.read_text())
        print(f"{r['nome']:28s} {'APROVADO' if r['aprovado'] else 'reprovado'}  2022 {r['falha_2022_dias']}  "
              f"sintética {r['sintetica_h']}", flush=True)
        if logger is not None:
            logger.report_scalar("aprovado", r["abordagem"], iteration=i, value=float(r["aprovado"]))
        linhas.append(r)
    resumo = pd.DataFrame(linhas); resumo.to_csv(out / "resumo.csv", index=False)
    print(f"\nresumo: {out / 'resumo.csv'}")
    if task is not None:
        logger.report_table("jobs", "resumo", iteration=0, table_plot=resumo)
        task.upload_artifact("resumo", artifact_object=out / "resumo.csv")
        for c in out.glob("*/cands"):
            shutil.rmtree(c)
        task.upload_artifact("janela", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
