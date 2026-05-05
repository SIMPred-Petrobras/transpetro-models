import os
from pathlib import Path
import pandas as pd
from transpetro_modelos.config import EQUIPMENT_CONFIGS

LOCAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "Dados"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _read_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".feather":
        return pd.read_feather(file_path)
    return pd.read_csv(file_path)


def load_equipment_data(equipment_id: str, from_clearml: bool = True) -> pd.DataFrame:
    """
    Carrega dados de um equipamento com DatetimeIndex.
    Se from_clearml=True, baixa do ClearML Dataset; caso contrário lê local.
    """
    config = EQUIPMENT_CONFIGS[equipment_id]

    if from_clearml:
        from clearml import Dataset
        ds = Dataset.get(
            dataset_name=config.dataset_name,
            dataset_project="Transpetro",
        )
        local_path = ds.get_local_copy()
        base = Path(local_path) / (config.dataset_file_stem or equipment_id)
        if base.with_suffix(".feather").exists():
            file_path = base.with_suffix(".feather")
        else:
            file_path = base.with_suffix(".csv")
    elif config.local_feather is not None:
        file_path = PROJECT_ROOT / config.local_feather
    else:
        base = LOCAL_DATA_DIR / (config.dataset_file_stem or equipment_id)
        if base.with_suffix(".feather").exists():
            file_path = base.with_suffix(".feather")
        else:
            file_path = base.with_suffix(".csv")

    df = _read_file(file_path)

    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif config.datetime_column is not None:
        df = df.set_index(config.datetime_column)
        df.index = pd.to_datetime(df.index)
    else:
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df
