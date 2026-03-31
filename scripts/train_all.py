import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS


def main(remote: bool = False) -> None:
    from scripts.train_equipment import main as train_one
    for equipment_id in EQUIPMENT_CONFIGS:
        print(f"\n{'='*50}")
        print(f"Training: {equipment_id}")
        print(f"{'='*50}")
        train_one(equipment_id, remote=remote)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    main(remote=args.remote)
