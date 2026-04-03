import shutil
import tempfile
from pathlib import Path
from clearml import Dataset
from transpetro_modelos.config import EQUIPMENT_CONFIGS

LOCAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "Dados"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def upload_equipment_dataset(equipment_id: str) -> None:
    config = EQUIPMENT_CONFIGS[equipment_id]

    # Use local_feather override if defined in config, otherwise default path
    if config.local_feather is not None:
        src_path = PROJECT_ROOT / config.local_feather
    else:
        src_path = LOCAL_DATA_DIR / f"{equipment_id}.feather"

    print(f"Uploading {equipment_id} ({src_path.stat().st_size / 1e6:.1f} MB) from {src_path.name}...")

    ds = Dataset.create(
        dataset_name=config.dataset_name,
        dataset_project="Transpetro",
    )

    # Ensure the file in the dataset has the standardized name {equipment_id}.feather
    # so that loading.py can find it regardless of the source filename.
    target_name = f"{equipment_id}.feather"
    if src_path.name == target_name:
        ds.add_files(str(src_path))
        ds.upload()
        ds.finalize()
    else:
        # Copy to a temp dir with the target name and upload from there.
        # ds.upload() must be called while the temp dir still exists.
        with tempfile.TemporaryDirectory() as tmp_dir:
            dst = Path(tmp_dir) / target_name
            shutil.copy2(src_path, dst)
            ds.add_files(str(dst))
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
    upload_metadata()
    print("\nAll datasets uploaded successfully.")


if __name__ == "__main__":
    main()
