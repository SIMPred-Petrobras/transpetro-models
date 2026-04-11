from pathlib import Path
from clearml import Dataset
from transpetro_modelos.config import EQUIPMENT_CONFIGS

LOCAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "Dados"


def upload_equipment_dataset(equipment_id: str) -> None:
    config = EQUIPMENT_CONFIGS[equipment_id]
    file_path = LOCAL_DATA_DIR / f"{equipment_id}.csv"

    print(f"Uploading {equipment_id} ({file_path.stat().st_size / 1e6:.1f} MB)...")

    ds = Dataset.create(
        dataset_name=config.dataset_name,
        dataset_project="Transpetro",
    )
    ds.add_files(str(file_path))
    ds.upload()
    ds.finalize()
    print(f"  Done: {config.dataset_name} (ID: {ds.id})")


def upload_metadata() -> None:
    file_path = LOCAL_DATA_DIR / "falhas.xlsx"
    print(f"Uploading metadata (falhas.xlsx)...")

    ds = Dataset.create(
        dataset_name="transpetro-metadata",
        dataset_project="Transpetro",
    )
    ds.add_files(str(file_path))
    ds.upload()
    ds.finalize()
    print(f"  Done: transpetro-metadata (ID: {ds.id})")


def main() -> None:
    for equipment_id in EQUIPMENT_CONFIGS:
        upload_equipment_dataset(equipment_id)
    print("\nAll datasets uploaded successfully.")


if __name__ == "__main__":
    main()
