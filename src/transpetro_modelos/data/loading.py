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
            dataset_project="Cabiunas brutos 2025-2026 alarmes mapeados",
        )

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
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df

def load_alarm_events(
    equipment_id: str,
    from_clearml: bool = True,
    alarm_filename: str = "alarmes_mapeados_colunas.csv",
    date_col: str = "data",
    group_gap_hours: float = 6.0,   # NOVO: junta alarmes a menos de X horas de distância
) -> list[pd.Timestamp]:
    """
    Carrega os eventos de alarme de um equipamento a partir do arquivo de
    alarmes (uma linha por sensor que disparou). Como vários sensores podem
    disparar quase juntos para o mesmo incidente de falha, os timestamps
    são agrupados: alarmes a menos de `group_gap_hours` de distância viram
    um único evento (usa o primeiro timestamp do grupo).
    """
    config = EQUIPMENT_CONFIGS[equipment_id]

    if from_clearml:
        from clearml import Dataset
        ds = Dataset.get(
            dataset_name=config.dataset_name,
            dataset_project="Cabiunas brutos 2025-2026 alarmes mapeados",
        )
        local_path = ds.get_local_copy()
        file_path = Path(local_path) / alarm_filename
    else:
        file_path = LOCAL_DATA_DIR / alarm_filename

    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])

    timestamps = sorted(df[date_col].tolist())
    if not timestamps:
        return []

    # ── Agrupamento: junta timestamps consecutivos próximos em um único evento ──
    grouped_events: list[pd.Timestamp] = [timestamps[0]]
    for ts in timestamps[1:]:
        gap = (ts - grouped_events[-1]) / pd.Timedelta(hours=1)
        if gap > group_gap_hours:
            grouped_events.append(ts)
        # senão, ts fica "absorvido" no evento anterior (não vira um novo evento)

    return grouped_events