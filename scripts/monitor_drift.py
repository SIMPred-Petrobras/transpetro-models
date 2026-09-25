"""Comando: `python scripts/monitor_drift.py --help`. O código está em src/transpetro_modelos/drift/monitor.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transpetro_modelos.drift.monitor import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
