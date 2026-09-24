"""
Compara os detectores da biblioteca (default_detectors) no benchmark de atraso de detecção
(detector/drift_benchmark.py), sobre o ERRO DE RECONSTRUÇÃO do B-8802B, em dois cenários:

  controle : modelo de produção (treino 2025); referência = erros de jan–jun/2025;
             série = jul–dez/2025, período sem drift → todo disparo é falso alarme
  drift    : modelo de 2022 lendo o dado de 2025–26; referência = erros do período normal
             de 2022; série a partir de 01/01/2025 (mudança conhecida) → atraso até detectar

Uso:  python scripts/compare_drift_detectors.py [--csv saida.csv]
Precisa dos dados locais do pacote de deploy (deploy_v2/Transpetro/*/dados, gitignorados)
e do CSV de inferência do modelo de 2022 sobre 2025–26 (gerado por monitor_drift / bench).
"""
import argparse, sys, warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DEP = ROOT / "deploy_v2/Transpetro"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(DEP))
import simpred_inference as si  # noqa: E402
from transpetro_modelos.detector.drift_detectors import default_detectors  # noqa: E402
from transpetro_modelos.detector.drift_benchmark import run_drift_benchmark  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    prod = pd.read_csv(DEP / "B-8802B-2025/scripts/b8802b2025_inferencia.csv", index_col=0, parse_dates=True)["reconstruction_error"]
    ref_c, serie_c = prod["2025-01-01":"2025-06-30"], prod["2025-07-01":"2025-12-31"]
    r_c = run_drift_benchmark(serie_c, default_detectors(ref_c.values), pd.Timestamp("2027-01-01"))

    b22 = next((DEP / "B-8802B/modelos").glob("model_*"))
    raw22 = si.carregar_dados(next((DEP / "B-8802B/dados").rglob("*_raw.csv")))
    e22 = si.prever(b22, si.carregar_modelo(b22), si.preprocessar(b22, raw22))["reconstruction_error"]
    old = pd.read_csv(ROOT / "results/monitor_drift/b8802b_modelo2022_em_2025-26_inferencia.csv",
                      index_col=0, parse_dates=True)["reconstruction_error"]
    r_d = run_drift_benchmark(old, default_detectors(e22[:"2022-06-20"].values), pd.Timestamp("2025-01-01"))

    out = (r_c.set_index("detector_name")[["n_false_alarms_before_drift"]]
           .rename(columns={"n_false_alarms_before_drift": "falsos_alarmes_controle (jul–dez/25)"})
           .join(r_d.set_index("detector_name")[["detected", "detection_delay_samples", "detection_delay_time"]]
                 .rename(columns={"detected": "detectou_drift", "detection_delay_samples": "atraso_amostras",
                                  "detection_delay_time": "atraso_tempo"})))
    out = out.sort_values(["falsos_alarmes_controle (jul–dez/25)", "atraso_amostras"])
    pd.set_option("display.width", 200)
    print(out.to_string())
    if a.csv: out.to_csv(a.csv)


if __name__ == "__main__":
    main()
