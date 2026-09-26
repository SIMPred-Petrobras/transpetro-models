"""
Retreino acumulativo após a mudança de conceito (B-8802B), com 3 candidatos por etapa.
Detalhes do experimento em src/transpetro_modelos/drift/acumulativo.py (rodar_candidatos).

    python scripts/experimento_acumulativo.py --remote --queue default     # roda no worker do ClearML
    python scripts/experimento_acumulativo.py --local                      # roda aqui, com os dados locais

No ClearML: métricas por etapa em Scalars (alarme no mês, falha sintética, falha de 2022, falso positivo
na validação por candidato), tabelas finais em Plots e os modelos/resultados como artifacts.
"""
import argparse, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", action="store_true", help="enfileira no ClearML e sai")
    ap.add_argument("--queue", default="default")
    ap.add_argument("--local", action="store_true", help="roda local, sem ClearML, com os dados locais")
    ap.add_argument("--sementes", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-meses", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--data-fixa", default="2026-06-10")
    ap.add_argument("--saida", default="results/acumulativo_b8802b_3c")
    ap.add_argument("--clearml-task-name", default="retreino-acumulativo-b8802b-3c")
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

    def reportar(k, linha, cands):
        if logger is None:
            return
        val = lambda v: float("nan") if v is None or pd.isna(v) else float(v)
        logger.report_scalar("alarme no mês em que operou (%)", "escolhido", iteration=k, value=val(linha["alarme_pct_vivo"]))
        logger.report_scalar("falha sintética no mês em que operou (h antes)", "100 %", iteration=k, value=val(linha["sintetica_100_h"]))
        logger.report_scalar("falha sintética no mês em que operou (h antes)", "50 %", iteration=k, value=val(linha["sintetica_50_h"]))
        logger.report_scalar("falha sintética na data fixa (h antes)", "100 % escolhido", iteration=k, value=val(linha["sintetica_fixa_100_h"]))
        logger.report_scalar("falha sintética na data fixa (h antes)", "50 % escolhido", iteration=k, value=val(linha["sintetica_fixa_50_h"]))
        logger.report_scalar("falha real 2022 (dias antes)", "escolhido", iteration=k, value=val(linha["falha_2022_dias"]))
        logger.report_scalar("alarme no normal de 2022 (%)", "escolhido", iteration=k, value=val(linha["normal_2022_pct"]))
        for c in cands:
            logger.report_scalar("falha sintética na data fixa (h antes)", f"100 % semente {c['semente']}", iteration=k, value=val(c["sintetica_fixa_100_h"]))
            logger.report_scalar("falha sintética usada na escolha (h antes)", f"semente {c['semente']}", iteration=k, value=val(c["sintetica_escolha_h"]))
            logger.report_scalar("falso positivo na validação (%)", f"semente {c['semente']}", iteration=k, value=val(c["fp_val_pct"]))

    saida = ROOT / a.saida
    res = ac.rodar_candidatos(saida, sementes=tuple(a.sementes), n_meses=a.n_meses, epochs=a.epochs, data_fixa=a.data_fixa,
                              from_clearml=not a.local, reportar=reportar, log=lambda m: print(m, flush=True))
    serie = ac.serie_emendada(saida)
    ep = ac.episodios(serie, saida)
    ep.to_csv(saida / "episodios.csv", index=False)
    fora = serie[serie["etapa"] > 0]
    print(f"fora da quarentena: {100 * fora['alarme'].mean():.3f} % do tempo em alarme; {len(ep)} episódios", flush=True)

    if task is not None:
        logger.report_single_value("alarme fora da quarentena (%)", 100 * float(fora["alarme"].mean()))
        logger.report_single_value("episódios de alarme", len(ep))
        logger.report_table("etapas", "escolhidos", iteration=0, table_plot=res)
        logger.report_table("candidatos", "todos", iteration=0, table_plot=pd.read_csv(saida / "candidatos.csv"))
        logger.report_table("episódios", "fora da quarentena", iteration=0, table_plot=ep)
        task.upload_artifact("etapas", artifact_object=saida / "etapas.csv")
        task.upload_artifact("candidatos", artifact_object=saida / "candidatos.csv")
        task.upload_artifact("episodios", artifact_object=saida / "episodios.csv")
        arq = shutil.make_archive(str(saida), "zip", saida)
        task.upload_artifact("resultados_completos", artifact_object=arq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
