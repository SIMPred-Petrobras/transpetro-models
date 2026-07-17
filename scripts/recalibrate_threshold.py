"""
Recalibra os thresholds de um bundle de deploy pela META DE FALSO POSITIVO.
=========================================================================

O threshold NÃO deve ser escolhido na mão. A meta de FP (quantos alarmes em
condição normal são toleráveis) é uma decisão de negócio; o threshold é o quantil
dos erros de reconstrução numa JANELA NORMAL que atinge essa meta:

    threshold_alarme   = quantil(erros_normais, 1 - fp_alarme)
    threshold_atencao  = quantil(erros_normais, 1 - fp_atencao)   # < alarme

Dois níveis evitam ofuscar o sinal fraco: "atenção" registra o desvio entre os
dois cortes (não aciona OS), "alarme" aciona. Os valores e os metadados da
calibração (janela, metas, nº de pontos, FP medido) são gravados no alarm.json.

Uso:
    uv run python scripts/recalibrate_threshold.py --equipment B-8802B \
        --normal-end 2022-06-30 --fp-alarm 0.01 --fp-attention 0.05 \
        --failure-date 2022-07-06

A janela normal = linhas ANTES de --normal-end (deixe uma margem antes da falha
para não contar a rampa de pré-falha como "normal"). Sem --normal-end, usa toda a
série (não recomendado se ela contém a falha).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "deploy"))

from simpred_inference import load_bundle, reconstruction_errors  # noqa: E402


def _find_bundle_dir(equipment: str, out_root: Path) -> Path:
    eq_dir = out_root / equipment
    matches = sorted((eq_dir / "modelos").glob("model_*"))
    if not matches:
        raise SystemExit(f"Nenhum bundle em {eq_dir / 'modelos'}. Empacote o equipamento antes.")
    return matches[-1]


def _find_data(bundle_dir: Path) -> Path:
    eq_dir = bundle_dir.parent.parent
    matches = list((eq_dir / "dados").rglob("*_raw.csv"))
    if not matches:
        raise SystemExit(f"Nenhum dado *_raw.csv em {eq_dir / 'dados'}.")
    return matches[0]


def _persist(flags: pd.Series, k: int, n: int) -> pd.Series:
    """Filtro de persistência k-de-n: dispara quando k dos últimos n pontos passam."""
    if n <= 1:
        return flags.astype(bool)
    return (flags.astype(int).rolling(n, min_periods=n).sum() >= k).fillna(False)


def recalibrate(
    bundle_dir: Path,
    data_file: Path,
    normal_end: str | None,
    method: str,
    fp_alarm: float,
    fp_attention: float,
    y_alarm: float,
    y_attention: float,
    persist_window: int | None,
    persist_min: int | None,
    failure_date: str | None,
) -> dict:
    bundle = load_bundle(bundle_dir)
    errors = reconstruction_errors(bundle, data_file)["reconstruction_error"]

    # janela normal = antes de normal_end (com margem antes da falha)
    normal = errors[errors.index < pd.to_datetime(normal_end)] if normal_end else errors
    if len(normal) < 100:
        raise SystemExit(
            f"Janela normal tem só {len(normal)} pontos — confira --normal-end ({normal_end})."
        )

    # persistência (k-de-n); sem --persist-*, cai no debounce do bundle (k=n consecutivos)
    if persist_window:
        n = int(persist_window)
        k = int(persist_min) if persist_min else n
    else:
        n = k = int(getattr(bundle, "debounce_consecutive", 1) or 1)

    mu, sd = float(normal.mean()), float(normal.std())
    if method == "sigma":
        # threshold = média + y·desvio-padrão dos erros na janela normal (régua interpretável)
        thr_alarm = mu + y_alarm * sd
        thr_attention = mu + y_attention * sd
    else:  # método "fp": quantil que atinge a meta de falso positivo
        thr_alarm = float(normal.quantile(1.0 - fp_alarm))
        thr_attention = float(normal.quantile(1.0 - fp_attention))
    if thr_attention >= thr_alarm:  # degenera; desativa o nível de atenção
        thr_attention = None

    # FP efetivo medido na janela normal, JÁ com a persistência (reflete o deploy)
    fp_alarm_meas = float(_persist(normal > thr_alarm, k, n).mean())
    fp_att_meas = (float(_persist(normal > thr_attention, k, n).mean())
                   if thr_attention is not None else None)

    report = {
        "old_threshold": float(bundle.threshold),
        "threshold": thr_alarm,
        "threshold_attention": thr_attention,
        "method": method,
        "mean_normal": mu,
        "std_normal": sd,
        "y_alarm": y_alarm if method == "sigma" else round((thr_alarm - mu) / (sd + 1e-12), 2),
        "y_attention": (y_attention if method == "sigma"
                        else (round((thr_attention - mu) / (sd + 1e-12), 2) if thr_attention is not None else None)),
        "fp_alarm_target": fp_alarm if method == "fp" else None,
        "fp_attention_target": fp_attention if method == "fp" else None,
        "fp_alarm_measured": round(fp_alarm_meas, 4),
        "fp_attention_measured": round(fp_att_meas, 4) if fp_att_meas is not None else None,
        "persistence": {"k": k, "n": n},
        "normal_window": {
            "start": str(normal.index.min()),
            "end": str(normal.index.max()),
            "n_points": int(len(normal)),
        },
    }

    # detecção na janela de pré-falha (persistência aplicada na série inteira, depois recortada)
    if failure_date:
        fd = pd.to_datetime(failure_date)
        start = pd.to_datetime(normal_end) if normal_end else errors.index.min()
        fl = _persist(errors > thr_alarm, k, n)
        pre_mask = (errors.index >= start) & (errors.index < fd)
        if pre_mask.sum():
            fl_pre = fl[pre_mask]
            crosses = fl_pre.index[fl_pre]
            report["prefailure"] = {
                "window": f"{start.date()} -> {fd.date()}",
                "detection_rate": round(float(fl_pre.mean()), 4),
                "first_alarm": str(crosses.min()) if len(crosses) else None,
            }

    # ── grava no alarm.json (preserva chaves existentes; registra a calibração) ──
    alarm_path = bundle_dir / "alarm.json"
    alarm = json.loads(alarm_path.read_text())
    alarm["threshold"] = thr_alarm
    alarm["threshold_attention"] = thr_attention
    if n > 1:
        alarm["debounce_window"] = n
        alarm["debounce_min"] = k
    alarm["threshold_calibration"] = {
        key: report[key] for key in (
            "method", "mean_normal", "std_normal", "y_alarm", "y_attention",
            "fp_alarm_target", "fp_attention_target", "fp_alarm_measured",
            "fp_attention_measured", "persistence", "normal_window",
        )
    }
    alarm_path.write_text(json.dumps(alarm, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="Recalibra thresholds de um bundle pela meta de FP.")
    ap.add_argument("--equipment", help="Chave do equipamento (localiza o bundle em deploy/Transpetro).")
    ap.add_argument("--bundle-dir", help="Caminho direto do bundle (alternativa a --equipment).")
    ap.add_argument("--data", help="CSV bruto (default: o *_raw.csv do próprio equipamento).")
    ap.add_argument("--normal-end", help="Fim da janela normal (YYYY-MM-DD); deixe margem antes da falha.")
    ap.add_argument("--method", choices=["fp", "sigma"], default="fp",
                    help="Régua do threshold: 'fp' (quantil por meta de FP) ou 'sigma' (média + y·desvio).")
    ap.add_argument("--fp-alarm", type=float, default=0.01, help="[fp] Meta de FP do alarme (default 0.01).")
    ap.add_argument("--fp-attention", type=float, default=0.05, help="[fp] Meta de FP da atenção (default 0.05).")
    ap.add_argument("--y-alarm", type=float, default=4.0, help="[sigma] nº de desvios p/ o alarme (default 4).")
    ap.add_argument("--y-attention", type=float, default=3.0, help="[sigma] nº de desvios p/ a atenção (default 3).")
    ap.add_argument("--persist-window", type=int, default=None, help="Persistência: janela n (k-de-n).")
    ap.add_argument("--persist-min", type=int, default=None, help="Persistência: mínimo k (default = n).")
    ap.add_argument("--failure-date", help="Data da falha conhecida (YYYY-MM-DD) p/ relatar detecção.")
    ap.add_argument("--out", default=None, help="Raiz dos bundles (default: deploy/Transpetro).")
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else PROJECT_ROOT / "deploy" / "Transpetro"
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
    elif args.equipment:
        bundle_dir = _find_bundle_dir(args.equipment, out_root)
    else:
        raise SystemExit("Informe --equipment ou --bundle-dir.")
    data_file = Path(args.data) if args.data else _find_data(bundle_dir)

    print(f"bundle: {bundle_dir}")
    print(f"dados : {data_file.name}")
    r = recalibrate(bundle_dir, data_file, args.normal_end, args.method,
                    args.fp_alarm, args.fp_attention, args.y_alarm, args.y_attention,
                    args.persist_window, args.persist_min, args.failure_date)

    print("\n── recalibração ──")
    p = r["persistence"]
    if r["method"] == "sigma":
        print(f"  método             : μ + y·σ  (μ={r['mean_normal']:.5f}, σ={r['std_normal']:.5f})")
        print(f"  threshold (alarme) : {r['old_threshold']:.4f} -> {r['threshold']:.4f}  "
              f"(y={r['y_alarm']}, FP medido {r['fp_alarm_measured']:.2%})")
        if r["threshold_attention"] is not None:
            print(f"  threshold (atenção): {r['threshold_attention']:.4f}  "
                  f"(y={r['y_attention']}, FP medido {r['fp_attention_measured']:.2%})")
    else:
        print(f"  threshold (alarme) : {r['old_threshold']:.4f} -> {r['threshold']:.4f}  "
              f"(meta {r['fp_alarm_target']:.0%} FP, medido {r['fp_alarm_measured']:.2%})")
        if r["threshold_attention"] is not None:
            print(f"  threshold (atenção): {r['threshold_attention']:.4f}  "
                  f"(meta {r['fp_attention_target']:.0%} FP, medido {r['fp_attention_measured']:.2%})")
    print(f"  persistência       : {p['k']}-de-{p['n']}")
    nw = r["normal_window"]
    print(f"  janela normal      : {nw['start']} -> {nw['end']}  ({nw['n_points']} pts)")
    if "prefailure" in r:
        pf = r["prefailure"]
        print(f"  pré-falha ({pf['window']}): detecção {pf['detection_rate']:.1%}, "
              f"1º alarme {pf['first_alarm']}")
    print("\n✓ alarm.json atualizado.")


if __name__ == "__main__":
    main()
