"""
Replay do retreino sobre o histórico: depois de uma mudança de conceito confirmada em --inicio, roda o pipeline de
retreino provisório todo mês (1, 2, … --meses meses do normal novo) como se estivesse em produção, e junta o resultado
de cada etapa (aprovado ou não, candidato, bateria) em resumo.csv. Usado para avaliar o pipeline de ponta a ponta em
notebooks/drift/replay_pipeline_B-8802B.ipynb.

    python scripts/replay_pipeline.py --equipment B-8802B-2025 --inicio 2025-01-06 --meses 12 \
        --out results/replay_pipeline --remote --queue default      # worker do ClearML (GPU)
    python scripts/replay_pipeline.py ... --baixar <task id>        # traz o resultado do ClearML para --out
    python scripts/replay_pipeline.py ... (sem --remote)            # roda aqui, com os dados locais

Cada etapa é `scripts/retrain_pipeline.py --provisorio`, com o mesmo --train-start e o --train-end avançando um mês.
Etapas já concluídas (resultado.json na pasta) são puladas; os candidatos treinados ficam em cache.

No worker, o dado de treino vem do Dataset do equipamento e os CSVs que a bateria lê (2025–26 e a falha de 2022) do
Dataset `transpetro-b-8802b-bateria-csv`, gravados nos mesmos caminhos do pacote de deploy. O resultado (bundles,
baterias, resumo) sobe como artifact `replay` e as métricas por etapa aparecem em Scalars.
"""
import argparse, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = "transpetro-b-8802b-bateria-csv"


def _baixar_csvs():
    from clearml import Dataset
    origem = Path(Dataset.get(dataset_name=DATASET_CSV, dataset_project="Transpetro").get_local_copy())
    for f in origem.rglob("*.csv"):
        destino = ROOT / f.relative_to(origem)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, destino)
        print(f"[dados] {destino.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # valores padrão (não obrigatórios): no worker do ClearML o script roda sem argumentos e lê os da task
    ap.add_argument("--equipment", default="B-8802B-2025")
    ap.add_argument("--inicio", default="2025-01-06", help="início do normal novo (data do disparo confirmado)")
    ap.add_argument("--meses", type=int, default=12)
    ap.add_argument("--out", default="results/replay_pipeline")
    ap.add_argument("--remote", action="store_true", help="enfileira no ClearML e sai")
    ap.add_argument("--queue", default="default")
    ap.add_argument("--baixar", default=None, help="id da task do ClearML: baixa o artifact `replay` para --out e sai")
    ap.add_argument("--clearml-task-name", default="replay-pipeline-b8802b")
    a = ap.parse_args()
    if a.baixar:
        from clearml import Task
        out = ROOT / a.out
        z = Task.get_task(task_id=a.baixar).artifacts["replay"].get_local_copy(extract_archive=False)
        out.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(z).extractall(out)
        print(f"resultado da task {a.baixar} extraído em {out}"); return 0

    task = logger = None
    no_worker = bool(os.environ.get("CLEARML_TASK_ID"))      # o agent do ClearML define esta variável
    if a.remote or no_worker:
        from clearml import Task
        Task.add_requirements("pyarrow"); Task.add_requirements("torch", package_version="")
        task = Task.init(project_name="Transpetro", task_name=a.clearml_task_name, output_uri=True, reuse_last_task_id=False)
        task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
        task.connect(vars(a))                           # no worker, preenche `a` com os parâmetros da task
        a.remote = True
        task.execute_remotely(queue_name=a.queue)       # daqui para baixo, só no worker
        logger = task.get_logger()
        _baixar_csvs()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ini = pd.Timestamp(a.inicio)
    linhas = []
    for k in range(1, a.meses + 1):
        fim = ini + pd.DateOffset(months=k)
        pasta = out / f"m{k:02d}"
        res_path = pasta / "resultado.json"
        if not res_path.exists():
            pasta.mkdir(exist_ok=True)
            cmd = [sys.executable, str(ROOT / "scripts/retrain_pipeline.py"), "--equipment", a.equipment, "--provisorio",
                   "--train-start", str(ini.date()), "--train-end", str(fim.date()), "--out", str(pasta),
                   "--operacao-confirmou"] + (["--from-clearml"] if a.remote else [])
            print(f"\n=== etapa {k}: {ini.date()} → {fim.date()} ===", flush=True)
            with open(pasta / "log.txt", "w") as log:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            bat = sorted(pasta.glob("battery_cand*.json"))
            ultimo = json.loads(bat[-1].read_text()) if bat else {}
            bundle = next(pasta.glob("model_*_VAE_provisorio"), None)
            cand = json.loads((bundle / "alarm.json").read_text())["threshold_calibration"]["candidate"] if bundle else None
            res_path.write_text(json.dumps({
                "etapa": k, "treino_ini": str(ini.date()), "treino_fim": str(fim.date()), "aprovado": rc == 0,
                "candidatos_testados": len(bat), "candidato": cand if rc == 0 else None,
                "bundle": str(bundle.relative_to(ROOT)) if (bundle and rc == 0) else None,
                "fp_val_pct": ultimo.get("fp_val_pct"),
                "falha_2022_dias": (ultimo.get("cross_era") or {}).get("lead_days"),
                "normal_2022_pct": (ultimo.get("cross_era") or {}).get("normal_rate_pct"),
                "sintetica_t0": (ultimo.get("synthetic") or {}).get("t0"),
                "sintetica_h": (ultimo.get("synthetic") or {}).get("100"),
            }, indent=1, ensure_ascii=False, default=str))
        r = json.loads(res_path.read_text())
        print(f"etapa {k:2d} ({r['treino_fim']}): {'APROVADO' if r['aprovado'] else 'reprovado'}  "
              f"candidato {r['candidato']}  2022 {r['falha_2022_dias']}  sintética {r['sintetica_h']}", flush=True)
        if logger is not None:
            val = lambda v: float("nan") if v is None else float(v)
            logger.report_scalar("aprovado", "etapa", iteration=k, value=float(r["aprovado"]))
            logger.report_scalar("falha real 2022 (dias antes)", "escolhido", iteration=k, value=val(r["falha_2022_dias"]))
            logger.report_scalar("falha sintética (h antes)", "escolhido", iteration=k, value=val(r["sintetica_h"]))
            logger.report_scalar("FP na validação (%)", "escolhido", iteration=k, value=val(r["fp_val_pct"]))
            logger.report_scalar("alarme no normal de 2022 (%)", "escolhido", iteration=k, value=val(r["normal_2022_pct"]))
        linhas.append(r)
    resumo = pd.DataFrame(linhas); resumo.to_csv(out / "resumo.csv", index=False)
    print(f"\nresumo: {out / 'resumo.csv'}")
    if task is not None:
        logger.report_table("etapas", "resumo", iteration=0, table_plot=resumo)
        task.upload_artifact("resumo", artifact_object=out / "resumo.csv")
        for c in out.glob("m*/cands"):           # candidatos não escolhidos não precisam subir
            shutil.rmtree(c)
        task.upload_artifact("replay", artifact_object=shutil.make_archive(str(out), "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
