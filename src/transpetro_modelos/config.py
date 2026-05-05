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
    dataset_file_stem: Optional[str] = None  # override filename stem when fetching from ClearML (default: equipment_id)
    val_start_date: Optional[datetime] = None  # fixed validation start date (e.g., Jul 1)


B4064A_NOVOS_PREPROCESS_PRESETS: dict[str, list[dict]] = {
    "baseline": [
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "moving_average": [
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "knn": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "moving_average_knn": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
}


EQUIPMENT_CONFIGS: dict[str, EquipmentConfig] = {
    "B-4064A_interpolated": EquipmentConfig(
        equipment_id="B-4064A",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a_interpolated",
        datetime_column=None,
        exclusion_days_before=10,
        val_start_date=datetime(2024, 8, 11),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 30},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=[
            {"step": "clip"},
            {"step": "normalize", "method": "standard"},
        ],
    ),
    "B-4064A-novos": EquipmentConfig(
        equipment_id="B-4064A-novos",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a-novos",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados-novos/B-4064A_novos.feather",
        val_start_date=datetime(2024, 7, 1),
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [-25.0]},
            {"step": "resample", "freq": "1h"},
            {"step": "ffill", "limit": 6},
            {"step": "filter_running", "column": "Corrente", "threshold": 5.0},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 0.0},
        ],
        preprocessing_steps=deepcopy(B4064A_NOVOS_PREPROCESS_PRESETS["baseline"]),
        preprocess_presets=deepcopy(B4064A_NOVOS_PREPROCESS_PRESETS),
    ),
    "B-8802B": EquipmentConfig(
        equipment_id="B-8802B",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 35.0},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "select_features", "features": ["Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA", "Vibração Bomba LNA", "Temperatura Bomba LA"]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-8802B-8s": EquipmentConfig(
        equipment_id="B-8802B-8s",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        dataset_file_stem="B-8802B",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 35.0},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-8802B-8s-nfr": EquipmentConfig(
        equipment_id="B-8802B-8s-nfr",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        dataset_file_stem="B-8802B",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-6511502A": EquipmentConfig(
        equipment_id="B-6511502A",
        failure_date=datetime(2023, 5, 15, 0, 0),
        failure_description="Quebra das lâminas do acoplamento",
        dataset_name="transpetro-b-6511502a",
        datetime_column=None,
        exclusion_days_before=10,
        local_feather="DadosV2/B-6511502A_pivoted.feather",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [32767.0]},
            {"step": "filter_running", "column": "CORRENTE ELÉTRICA DO MOTOR", "threshold": 60.0},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "select_features", "features": [
                "CORRENTE ELÉTRICA DO MOTOR",
                "PRESSÃO SUCÇÃO",
                "PRESSÃO DESCARGA",
                "DESLOC. AXIAL EIXO BB LNA 1 ZE-50",
                "DESLOC. AXIAL EIXO BB LNA 2 ZE-51",
                "TEMP. MANCAL LNA MOT TE-07A1/A2",
                "VIB. MANCAL RADIAL BB LA 0° VE-50C",
                "VIB. MANCAL RADIAL BB LA 90° VE-51C",
                "VIB. MANCAL RADIAL BB LNA 0° VE-50D",
                "VIB. MANCAL RADIAL BB LNA 90° VE-51D",
                "VIB. MANCAL RADIAL MOT LA 0° VE-51B",
                "VIB. MANCAL RADIAL MOT LA 90° VE-50B",
                "VIB. MANCAL RADIAL MOT LNA 0° VE-50A",
                "VIB. MANCAL RADIAL MOT LNA 90° VE-51A",
            ]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-90001A": EquipmentConfig(
        equipment_id="B-90001A",
        failure_date=datetime(2021, 8, 28, 0, 0),
        failure_description="Afrouxamento no aperto dos parafusos do mancal do lado acoplado da bomba",
        dataset_name="transpetro-b-90001a",
        datetime_column=None,
        exclusion_days_before=10,
        preprocessing_steps=[
            {"step": "normalize", "method": "standard"},
        ],
    ),
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
