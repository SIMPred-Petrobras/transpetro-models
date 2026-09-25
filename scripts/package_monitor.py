"""
Gera o monitor autocontido do pacote de deploy (deploy_v2/Transpetro/monitor_drift.py) a partir de
src/transpetro_modelos/drift/monitor.py, embutindo as classes do detector (BaseDriftDetector e
CalibratedKSDetector de drift/detectors.py). O arquivo gerado não depende da lib interna: só
pandas/numpy/scipy e o simpred_inference.py do próprio pacote.

    python scripts/package_monitor.py            # regenera
    python scripts/package_monitor.py --check    # falha (exit 1) se a cópia do deploy estiver desatualizada

Rodar depois de qualquer mudança em drift/monitor.py ou no CalibratedKSDetector.
"""
import argparse, ast, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIFT = ROOT / "src/transpetro_modelos/drift"
OUT = ROOT / "deploy_v2/Transpetro/monitor_drift.py"
IMPORT_LINE = "from transpetro_modelos.drift.detectors import CalibratedKSDetector"
CLASSES = ("BaseDriftDetector", "CalibratedKSDetector")


def build() -> str:
    det_src = (DRIFT / "detectors.py").read_text()
    tree = ast.parse(det_src)
    blocos = {n.name: ast.get_source_segment(det_src, n) for n in tree.body if isinstance(n, ast.ClassDef)}
    faltando = [c for c in CLASSES if c not in blocos]
    if faltando:
        raise SystemExit(f"classes não encontradas em detectors.py: {faltando}")
    embutido = "\n".join([
        "from abc import ABC, abstractmethod",
        "from scipy import stats",
        "",
        "",
        "# ── detector embutido (cópia GERADA de src/transpetro_modelos/drift/detectors.py) ──",
        "",
    ] + [blocos[c] + "\n\n" for c in CLASSES])

    mon_src = (DRIFT / "monitor.py").read_text()
    if mon_src.count(IMPORT_LINE) != 1:
        raise SystemExit(f"monitor.py precisa ter exatamente uma linha '{IMPORT_LINE}'")
    corpo = mon_src.replace(IMPORT_LINE, embutido)
    cabecalho = ("# ARQUIVO GERADO por scripts/package_monitor.py a partir de src/transpetro_modelos/drift/monitor.py\n"
                 "# e drift/detectors.py. Não edite aqui: edite a origem e rode o gerador.\n")
    return cabecalho + corpo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    novo = build()
    if a.check:
        atual = OUT.read_text() if OUT.exists() else ""
        if atual != novo:
            print(f"DESATUALIZADO: {OUT.relative_to(ROOT)} — rode python scripts/package_monitor.py")
            return 1
        print(f"ok: {OUT.relative_to(ROOT)} está em dia")
        return 0
    OUT.write_text(novo)
    print(f"gerado: {OUT.relative_to(ROOT)} ({len(novo.splitlines())} linhas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
