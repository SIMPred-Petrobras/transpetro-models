import os
from pathlib import Path
import pandas as pd
from transpetro_modelos.config import EQUIPMENT_CONFIGS

LOCAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "Dados"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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
        print(ds)
        local_path = ds.get_local_copy()
        file_path = Path(local_path) / f"{equipment_id}.csv"
    elif config.local_feather is not None:
        file_path = PROJECT_ROOT / config.local_feather
    else:
        file_path = LOCAL_DATA_DIR / f"{equipment_id}.csv"

    df = pd.read_csv(file_path)

    if config.datetime_column is not None:
        df = df.set_index(config.datetime_column)
        df.index = pd.to_datetime(df.index)
    else:
        df = df.set_index("Timestamp")
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df
