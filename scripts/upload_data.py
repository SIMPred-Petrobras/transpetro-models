import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transpetro_modelos.config import EQUIPMENT_CONFIGS
from transpetro_modelos.data.upload_datasets import upload_equipment_dataset, upload_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--equipment",
        choices=list(EQUIPMENT_CONFIGS.keys()),
        help="Upload only this equipment (default: all + metadata)",
    )
    args = parser.parse_args()

    if args.equipment:
        upload_equipment_dataset(args.equipment)
    else:
        for equipment_id in EQUIPMENT_CONFIGS:
            upload_equipment_dataset(equipment_id)
        print("\nAll datasets uploaded successfully.")
