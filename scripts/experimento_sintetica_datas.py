"""
Reavalia os modelos do retreino acumulativo (B-8802B) com a falha sintética injetada em várias datas,
sem treinar nada. Testa se a baixa sensibilidade medida em 10/06/2026 vem dos modelos ou da deriva de
temperatura daquele período (mancal LA ~8 °C abaixo do normal de 2025).

Os modelos vêm do artifact `resultados_completos` da task de origem (retreino-acumulativo-b8802b-3c).

    python scripts/experimento_sintetica_datas.py --remote --queue default
    python scripts/experimento_sintetica_datas.py --local --saida results/acumulativo_b8802b_3c   # modelos já locais
"""
import argparse, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# todas em 2026 (nenhuma etapa treinou com 2026): jan–mar antes da deriva de temperatura, jun–jul durante
DATAS = ["2026-01-20", "2026-02-10", "2026-03-10", "2026-06-10", "2026-07-15"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", action="store_true"); ap.add_argument("--queue", default="default")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--task-origem", default="2a0ff0545789427db1c7e89e817c0cb7")
    ap.add_argument("--datas", nargs="+", default=DATAS)
    ap.add_argument("--saida", default="results/acumulativo_b8802b_3c")
    ap.add_argument("--clearml-task-name", default="sintetica-varias-datas-b8802b")
    a = ap.parse_args()

    task = logger = None
    if not a.local:
        from clearml import Task
        Task.add_requirements("pyarrow")
        Task.add_requirements("torch", package_version="")
        task = Task.init(project_name="Transpetro", task_name=a.clearml_task_name, output_uri=True, reuse_last_task_id=False)
        task.set_base_docker("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
        task.connect(vars(a))
        if a.remote:
            task.execute_remotely(queue_name=a.queue)
        logger = task.get_logger()

    import pandas as pd
    from transpetro_modelos.drift import acumulativo as ac

    saida = ROOT / a.saida
    if not a.local:   # baixa os modelos treinados da task de origem
        from clearml import Task
        origem = Task.get_task(task_id=a.task_origem)
        zip_path = origem.artifacts["resultados_completos"].get_local_copy(extract_archive=False)
        saida.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(zip_path).extractall(saida)
        print(f"modelos da task {a.task_origem} extraídos em {saida}", flush=True)

    df = ac.sintetica_varias_datas(saida, a.datas, from_clearml=not a.local, log=lambda m: print(m, flush=True))
    df.to_csv(saida / "sintetica_varias_datas.csv", index=False)

    detecta = df.assign(ok=df["sintetica_100_h"].fillna(-999) >= 20)
    resumo = detecta.groupby(["data", "injecao_em", "temp_LA_mediana"]).agg(
        candidatos=("ok", "size"), detectam_20h=("ok", "sum"),
        mediana_h=("sintetica_100_h", "median")).reset_index()
    resumo["pct_detectam_20h"] = 100 * resumo["detectam_20h"] / resumo["candidatos"]
    print(resumo.round(1).to_string(index=False), flush=True)
    resumo.to_csv(saida / "sintetica_varias_datas_resumo.csv", index=False)

    if task is not None:
        val = lambda v: float("nan") if v is None or pd.isna(v) else float(v)
        for r in df.itertuples():
            logger.report_scalar(f"sintética 100 % (h antes) — {r.data}", f"semente {r.semente}", iteration=r.etapa, value=val(r.sintetica_100_h))
            if r.escolhido:
                logger.report_scalar("sintética 100 % do escolhido (h antes)", r.data, iteration=r.etapa, value=val(r.sintetica_100_h))
        for r in resumo.itertuples():
            logger.report_single_value(f"% candidatos ≥ 20 h — {r.data} (temp LA {r.temp_LA_mediana:.1f} °C)", r.pct_detectam_20h)
        logger.report_table("resumo por data", "todas as etapas", iteration=0, table_plot=resumo)
        logger.report_table("todos os candidatos", "por data", iteration=0, table_plot=df)
        task.upload_artifact("sintetica_varias_datas", artifact_object=saida / "sintetica_varias_datas.csv")
        task.upload_artifact("resumo", artifact_object=saida / "sintetica_varias_datas_resumo.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
