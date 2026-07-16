"""
SIMPred — Inferência de anomalias do B-3403C.

Script FINO: só aponta o bundle deste equipamento e executa os 4 passos, todos
implementados no módulo compartilhado `simpred_inference.py` (na pasta Transpetro/).
Não depende de nenhuma biblioteca interna nossa.

    python3 b3403c_exemplo.py
"""
import sys
from pathlib import Path

# torna o módulo compartilhado (Transpetro/simpred_inference.py) importável
_RAIZ = Path(__file__).resolve().parents[2]   # .../deploy_v2/Transpetro
sys.path.insert(0, str(_RAIZ))
import simpred_inference as si

# ── Bundle e dados deste equipamento (tudo relativo a esta pasta) ────────────
EQUIP = "B-3403C"
EQUIP_DIR = Path(__file__).resolve().parent.parent      # .../Transpetro/B-3403C
BUNDLE_DIR = EQUIP_DIR / "modelos" / "model_2023-01-01_2023-09-15_DENSE"
DADOS_DIR = EQUIP_DIR / "dados"
CSV_NOME = "data_2023-01-01_2023-09-15_raw.csv"


def main():
    print(f"[{EQUIP}] 1/4 carregando dados...")
    csv = si.achar_csv(DADOS_DIR, CSV_NOME)
    df = si.carregar_dados(csv)
    print(f"        {len(df)} linhas brutas")

    print(f"[{EQUIP}] 2/4 pre-processando...")
    df_proc = si.preprocessar(BUNDLE_DIR, df)
    print(f"        {len(df_proc)} instantes, {df_proc.shape[1]} features")

    print(f"[{EQUIP}] 3/4 carregando modelo...")
    model = si.carregar_modelo(BUNDLE_DIR)

    print(f"[{EQUIP}] 4/4 inferindo...")
    res = si.prever(BUNDLE_DIR, model, df_proc)

    # ── Resumo (sem despejar todos os alarmes na tela) ──
    n_alarme = int((res["severity"] == "alarme").sum())
    n_atencao = int((res["severity"] == "atencao").sum())
    alarmes = res[res["is_anomaly"]]
    print(f"\n{EQUIP}: {len(res)} instantes | {n_atencao} atenção | {n_alarme} alarme")
    if len(alarmes):
        print(f"        1º alarme: {alarmes.index.min()}  |  último: {alarmes.index.max()}")

    # Salva o resultado completo (para inspeção / integração)
    nome_csv = EQUIP.lower().replace("-", "").replace(".", "") + "_inferencia.csv"
    saida = Path(__file__).parent / nome_csv
    res.to_csv(saida)
    print(f"        resultado salvo em: {saida.name}")


if __name__ == "__main__":
    main()
