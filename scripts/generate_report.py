"""
Gera o RELATÓRIO DE RESULTADO de um equipamento (Markdown) para o time de engenharia
integrar — vai dentro de deploy/Transpetro/<EQ>/documentos/.

Puxa tudo do bundle empacotado + inferência sobre os dados do próprio equipamento:
thresholds (alarm.json), métricas de treino, FP na janela normal, lead time na falha,
distribuição de severidade, contrato do bundle e como chamar a inferência. Inclui as
ressalvas conhecidas (estado desligado a filtrar, mudança de regime).

Uso:
    uv run python scripts/generate_report.py --equipment B-8802B
    uv run python scripts/generate_report.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "src"))

from simpred_inference import load_bundle, predict  # noqa: E402

# Janelas (falha + fim da janela normal) e ressalvas por equipamento.
META = {
    "B-8802B": {
        "falha": "2022-07-06", "normal_end": "2022-06-16",
        "desc": "Falha em bomba (descrição a confirmar com a manutenção)",
        "off_sensor": "Pressão Descarga", "off_frac": 0.431, "off_cut": 12.9,
        "regime": None,
    },
    "B-6511502A": {
        "falha": "2023-05-15", "normal_end": "2023-04-25",
        "desc": "Falha em bomba (descrição a confirmar com a manutenção)",
        "off_sensor": "CORRENTE ELÉTRICA DO MOTOR", "off_frac": 0.658, "off_cut": 108.0,
        "regime": "Tensões/correntes de linha só entram em operação após 2022-08-22 "
                  "(onset de instrumentação) — iniciar a janela normal depois dessa data.",
    },
    "B-4064A": {
        "falha": "2024-08-30", "normal_end": "2024-08-10",
        "desc": "Roçamento interno do rotor com a carcaça da bomba",
        "off_sensor": "Corrente", "off_frac": 0.37, "off_cut": 2.55,
        "regime": "ATENÇÃO: mudança de regime térmico pós-reparo — Temp. Bomba LNA sobe +21°C "
                  "em 2025. O bundle atual é calibrado em 2024 e gera FALSO POSITIVO permanente "
                  "em 2025+. Antes de operar em dados novos (2025+), re-baselinar no regime saudável "
                  "de 2025 (ou usar resíduo Temp~Corrente).",
    },
}


def _bundle_dir(eq: str) -> Path:
    d = ROOT / "deploy" / "Transpetro" / eq
    return next((d / "modelos").glob("model_*"))


def build_report(eq: str) -> Path:
    if eq not in META:
        raise SystemExit(f"Sem metadados para {eq}. Adicione em META.")
    m = META[eq]
    eq_dir = ROOT / "deploy" / "Transpetro" / eq
    bdir = _bundle_dir(eq)
    bundle = load_bundle(bdir)
    alarm = json.loads((bdir / "alarm.json").read_text())
    data_file = next((eq_dir / "dados").rglob("*_raw.csv"))

    r = predict(bundle, data_file)
    fd = pd.Timestamp(m["falha"])
    ne = pd.Timestamp(m["normal_end"])
    normal = r[r.index < ne]
    pre = r[(r.index >= ne) & (r.index < fd)]
    al_pre = pre[pre["is_anomaly"]]
    first = al_pre.index.min() if len(al_pre) else None
    lead = (fd - first) if first is not None else None
    sev = r["severity"].value_counts().to_dict()
    feats = alarm.get("features", [])
    cal = alarm.get("threshold_calibration", {})

    lines = []
    A = lines.append
    A(f"# Relatório de resultado — {eq}\n")
    A(f"> Gerado a partir do bundle de deploy. Material de apoio para a integração pelo time de engenharia.\n")

    A("## 1. Resumo\n")
    A(f"- **Equipamento:** {eq}")
    A(f"- **Falha conhecida:** {m['falha']} — {m['desc']}")
    A(f"- **Modelo:** {bundle.model_type.upper()} (autoencoder multivariado)")
    A(f"- **Sensores usados ({len(feats)}):** {', '.join(feats) if feats else '— ver pipeline.json'}")
    A(f"- **FP na janela normal (< {m['normal_end']}):** {100*normal['is_anomaly'].mean():.2f}% (meta 1%)")
    if first is not None:
        A(f"- **1º alarme antes da falha:** {str(first)[:16]} → **lead time ≈ {str(lead).split(',')[0]}**")
    else:
        A(f"- **Detecção pré-falha:** nenhum alarme na janela pré-falha")
    A(f"- **Severidade na série:** " + " · ".join(f"{k}={v}" for k, v in sev.items()) + "\n")

    A("## 2. Thresholds (alarm.json)\n")
    A(f"- **Alarme:** {bundle.threshold:.4f}")
    A(f"- **Atenção:** {bundle.threshold_attention:.4f}" if bundle.threshold_attention else "- **Atenção:** (não definido)")
    A(f"- **Debounce:** {bundle.debounce_consecutive} pontos consecutivos")
    if cal:
        A(f"- **Calibração:** meta FP alarme {cal.get('fp_alarm_target')} (medido {cal.get('fp_alarm_measured')}), "
          f"atenção {cal.get('fp_attention_target')} (medido {cal.get('fp_attention_measured')})")
        nw = cal.get("normal_window", {})
        if nw:
            A(f"  - janela normal de calibração: {str(nw.get('start'))[:10]} → {str(nw.get('end'))[:10]} ({nw.get('n_points')} pts)")
    A("")

    A("## 3. Como integrar\n")
    A("Conteúdo do bundle (`modelos/model_*/`):\n")
    A("| Arquivo | Conteúdo |")
    A("|---|---|")
    A("| `model.pt` | modelo PyTorch (módulo inteiro) |")
    A("| `preprocessing.pkl` | scaler/clip/coefs ajustados no treino (reuso, sem refit) |")
    A("| `pipeline.json` | passos de pré-processamento congelados |")
    A("| `alarm.json` | model_type, thresholds (alarme/atenção), debounce, features, métricas |\n")
    A("**Dependências (instalar uma vez):**\n")
    A("```bash")
    A("# o pacote transpetro_modelos é a fonte do preprocessing e das classes do modelo")
    A("pip install transpetro_modelos-0.1.0-py3-none-any.whl   # wheel incluído em scripts/")
    A("# (puxa torch, pandas, scikit-learn, pyarrow automaticamente)")
    A("```")
    A("- O `simpred_inference.py` **acompanha o bundle** em `scripts/` (não precisa de mais nada do repo).\n")
    A("**Inferência:**\n")
    A("```python")
    A("import sys; sys.path.insert(0, 'scripts')   # onde está o simpred_inference.py")
    A("from simpred_inference import load_bundle, predict")
    A(f"bundle = load_bundle('modelos/{bdir.name}')")
    A("result = predict(bundle, 'dados/.../data_..._raw.csv')  # csv ou DataFrame")
    A("# result: DataFrame [reconstruction_error, is_anomaly, severity] por timestamp")
    A("```")
    A("- **Saída:** `reconstruction_error` (float), `severity` ∈ {normal, atencao, alarme}, "
      "`is_anomaly` (=alarme, já com debounce).")
    A(f"- **Regra:** alarme quando o erro supera {bundle.threshold:.3f} por {bundle.debounce_consecutive} "
      "pontos consecutivos; atenção é um nível intermediário (registra, não aciona OS).")
    A("- **Entrada:** os mesmos sensores da seção 1, em colunas; índice temporal. O preprocessing "
      "(scaler/clip/resample) é reaplicado internamente a partir do `preprocessing.pkl`.")
    A(f"- **Exemplo pronto:** `scripts/{eq.lower().replace('.','').replace('-','')}-exemplo.py`\n")

    A("## 4. Ressalvas para a integração\n")
    A(f"- **Estado desligado:** ~{100*m['off_frac']:.0f}% das amostras são equipamento parado "
      f"(`{m['off_sensor']}` ≈ 0). Recomenda-se **filtrar** (`{m['off_sensor']}` > {m['off_cut']}) "
      f"antes de pontuar/treinar — o estado parado polui a baseline.")
    if m["regime"]:
        A(f"- **Mudança de regime:** {m['regime']}")
    else:
        A("- **Regime estável:** sem deslocamento de patamar relevante no período analisado.")
    A("")

    out = eq_dir / "documentos" / f"RELATORIO_{eq}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Gera relatório de resultado por equipamento (Markdown).")
    ap.add_argument("--equipment", choices=list(META.keys()))
    ap.add_argument("--all", action="store_true", help="Gera para todos os equipamentos empacotados.")
    args = ap.parse_args()
    eqs = list(META.keys()) if args.all else [args.equipment]
    if not eqs or eqs == [None]:
        raise SystemExit("Informe --equipment ou --all.")
    for eq in eqs:
        if not (ROOT / "deploy" / "Transpetro" / eq).exists():
            print(f"  [skip] {eq} não empacotado.")
            continue
        out = build_report(eq)
        print(f"  ✓ {eq}: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
