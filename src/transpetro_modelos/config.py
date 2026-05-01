from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class EquipmentConfig:
    equipment_id: str
    failure_date: datetime
    failure_description: str
    dataset_name: str
    datetime_column: Optional[str]  # "Data Hora" para B-402E, "Timestamp" para B-4064A novos, None para demais
    exclusion_days_before: int
    preprocessing_steps: list[dict]
    pre_split_steps: list[dict] = field(default_factory=list)  # resample, filter_running (roda antes do split)
    preprocess_presets: dict[str, list[dict]] = field(default_factory=dict)
    local_feather: Optional[str] = None  # override path for local loading (relative to project root)
    val_start_date: Optional[datetime] = None  # fixed validation start date (e.g., Jul 1)
    val_end_date: Optional[datetime] = None


PREPROCESSING_PIPELINES:dict[str, list[dict]] = {
    "baseline": [
        {"step": "clip"},
        {"step": "normalize", "method": "standard"},
    ],

    "knn": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "standard"},
    ],

    "moving_average": [
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "standard"},
    ],
}


EQUIPMENT_CONFIGS: dict[str, EquipmentConfig] = {
    "B-4064A": EquipmentConfig(
        equipment_id="B-4064A",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-4064A.csv",
        val_start_date=datetime(2024, 5, 1),
        val_end_date=datetime(2024, 5, 31),
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [-25.0]},
            {"step": "resample", "freq": "1h"},
            {"step": "interpolate", "method": "time", "limit": 4},
            {"step": "filter_running", "column": "B-4064A: Corrente", "threshold": 1.0},
            {"step": "filter_running", "column": "B-4064A: Pressão Descarga", "threshold": 0.0},
            {"step": "filter_running", "column": "B-4064A: Pressão Sucção", "threshold": 0.0},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline"]),
        preprocess_presets=deepcopy(PREPROCESSING_PIPELINES)
    ),

    "B-4064A_interpolated": EquipmentConfig(
        equipment_id="B-4064A_interpolated",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-4064A_interpolated.csv",
        val_start_date=datetime(2024, 8, 11),
        val_end_date=datetime(2024, 8, 20),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 30},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline"]),
        preprocess_presets=deepcopy(PREPROCESSING_PIPELINES)
    )
}


def get_preprocessing_steps(equipment_id: str, preset: str = "baseline") -> list[dict]:
    config = EQUIPMENT_CONFIGS[equipment_id]
    if config.preprocess_presets:
        if preset not in config.preprocess_presets:
            available = ", ".join(sorted(config.preprocess_presets))
            raise ValueError(f"Unknown preprocess preset '{preset}' for {equipment_id}. Available: {available}")
        return deepcopy(config.preprocess_presets[preset])

    if preset != "baseline":
        raise ValueError(f"Equipment {equipment_id} only supports preprocess_preset='baseline'")

    return deepcopy(config.preprocessing_steps)
