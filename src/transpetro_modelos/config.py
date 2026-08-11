from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class EquipmentConfig:
    equipment_id: str
    dataset_name: str
    datetime_column: Optional[str]
    exclusion_days_before: int
    preprocessing_steps: list[dict]

    failure_date: datetime | None = None
    failure_description: str | None = None
    failure_events: list[str] | None = None
    failure_descriptions: list[str] | None = None

    pre_split_steps: list[dict] = field(default_factory=list)
    preprocess_presets: dict[str, list[dict]] = field(default_factory=dict)
    local_feather: Optional[str] = None
    val_start_date: Optional[datetime] = None
    val_end_date: Optional[datetime] = None     


COMMUM_PREPROCESSING_STEPS: list[dict] = [
    {"step": "filter_running", "column": "B-4064A: Corrente", "threshold": 1.0},
    {"step": "filter_running", "column": "B-4064A: Pressão Descarga", "threshold": 0.0},
    {"step": "filter_running", "column": "B-4064A: Pressão Sucção", "threshold": 0.0},
]

PREPROCESSING_PIPELINES:dict[str, list[dict]] = {
    "baseline_raw": [
        {"step": "interpolate", "method": "time", "limit": 4},
        *COMMUM_PREPROCESSING_STEPS,
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "baseline_raw_no_common": [
        {"step": "interpolate", "method": "time", "limit": 4},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "baseline_interpolated": [
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "moving_average_raw": [
        {"step": "interpolate", "method": "time", "limit": 4},
        *COMMUM_PREPROCESSING_STEPS,
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "moving_average_raw_no_common": [
        {"step": "interpolate", "method": "time", "limit": 4},
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "moving_average_interpolated": [
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "knn_raw": [
        {"step": "interpolate", "method": "time", "limit": 4},
        *COMMUM_PREPROCESSING_STEPS,
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "knn_raw_no_common": [
        {"step": "interpolate", "method": "time", "limit": 4},
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],

    "knn_interpolated": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
}

RAW_PRESETS = {
    k: deepcopy(v)
    for k, v in PREPROCESSING_PIPELINES.items()
    if k.endswith("_raw")
}

RAW_NO_COMMON_PRESETS = {
    k: deepcopy(v)
    for k, v in PREPROCESSING_PIPELINES.items()
    if k.endswith("_no_common")
}

INTERPOLATED_PRESETS = {
    k: deepcopy(v)
    for k, v in PREPROCESSING_PIPELINES.items()
    if k.endswith("_interpolated")
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
            {"step": "resample", "freq": "1h"}
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_raw"]),
        preprocess_presets=RAW_PRESETS,
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
            {"step": "remove_transients", "minutes": 10, "gap_minutes": 30},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_interpolated"]),
        preprocess_presets=INTERPOLATED_PRESETS,
    ),

    "B-3403C_interpolated": EquipmentConfig(
        equipment_id="B-3403C_interpolated",
        failure_date=datetime(2023, 9, 12),
        failure_description="Quebra da ponta do eixo LNA da bomba",
        dataset_name="transpetro-b-3403c_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-3403C_interpolated.csv",
        val_start_date=datetime(2023, 8, 24),
        val_end_date=datetime(2023, 9, 2),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 1},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_interpolated"]),
        preprocess_presets=INTERPOLATED_PRESETS,
    ),

    "B-90001A_interpolated": EquipmentConfig(
        equipment_id="B-90001A_interpolated",
        failure_date=datetime(2021, 8, 28),
        failure_description="Afrouxamento no aperto dos parafusos do mancal do lado acoplado da bomba",
        dataset_name="transpetro-b-90001a_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-90001A_interpolated.csv",
        val_start_date=datetime(2021, 8, 9),
        val_end_date=datetime(2021, 8, 18),
        pre_split_steps=[
            {"step": "filter_threshold", "columns": ['Vibração Motor LNA Y', 'Vibração Motor LA X', 'Vibração Motor LA Y', 'Vibração Bomba LA X', 'Vibração Bomba LA Y', 'Vibração Bomba LNA X', 'Vibração Bomba LNA Y'], "threshold": 10},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_interpolated"]),
        preprocess_presets=INTERPOLATED_PRESETS,
    ),

    "B-24001B_interpolated": EquipmentConfig(
        equipment_id="B-24001B_interpolated",
        failure_date=datetime(2025, 1, 6),
        failure_description="Vibração elevada mancal LNA da bomba",
        dataset_name="transpetro-b-24001b_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-24001B_interpolated.csv",
        val_start_date=datetime(2024, 11, 27),
        val_end_date=datetime(2024, 12, 27),
        pre_split_steps=[
            {"step": "filter_threshold", "columns": ['VIBRAÇÃO DO MANCAL BOMBA LA', 'VIBRAÇÃO DO MANCAL BOMBA LNA ', 'VIBRAÇÃO DO MANCAL MOTOR LA (003)', 'VIBRAÇÃO DO MANCAL MOTOR LA (004)', 'VIBRAÇÃO DO MANCAL MOTOR LNA (006)'], "threshold": "otsu", "mode": "all", "fixed_thresholds": { 'VIBRAÇÃO DO MANCAL MOTOR LNA (005)': 9}},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_interpolated"]),
        preprocess_presets=INTERPOLATED_PRESETS,
    ),

    "B-8801C_interpolated": EquipmentConfig(
        equipment_id="B-8801C_interpolated",
        failure_date=datetime(2024, 7, 5),
        failure_description="Vibração elevada mancal LA motor e bomba",
        dataset_name="transpetro-b-8801c_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-8801C_interpolated.csv",
        val_start_date=datetime(2024, 5, 1),
        val_end_date=datetime(2024, 6, 25),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 1},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_interpolated"]),
        preprocess_presets=INTERPOLATED_PRESETS,    
    ),

    "B-5501B_interpolated": EquipmentConfig(
        equipment_id="B-5501B_interpolated",
        failure_events=["2022-10", "2023-10", "2024-01"],
        failure_descriptions=["Sobreaquecimento do mancal LA do motor", "Desgaste mancal LA da bomba", "Desgaste mancal LNA da bomba"],
        dataset_name="transpetro-b-5501b_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-5501B_interpolated.csv",
        val_start_date=datetime(2022, 3, 1),
        val_end_date=datetime(2022, 4, 1),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 1},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_interpolated"]),
        preprocess_presets=INTERPOLATED_PRESETS,
    ),

    "cabiunas_2025_2026": EquipmentConfig(
        equipment_id="cabiunas_2025_2026",
        dataset_name="Cabiunas brutos 2025-2026 alarmes mapeados",
        datetime_column="data_datetime",
        val_start_date=datetime(2025, 1, 1),
        val_end_date=datetime(2025, 1, 10),
        exclusion_days_before=10,
        pre_split_steps=[
            {"step": "filter_running", "column": "RUNNING_A", "threshold": 1},
            {"step": "remove_transients", "minutes": 10},
            {"step": "select_features", "features": ['data_datetime', '954005_624_TI_0325', '954005_624_PI_0315', '954005_624_PI_0319', '954005_624_PI_0340', '954005_624_PI_0339', '954005_624_PDI_0317', 'TC382_03_A', 'T5_AVG_A', 'TC382_02_A', '954005_624_PDI_0302', 'TC382_05_A', '954005_624_TI_0315', '954005_624_TI_0317', 'TC382_06_A', 'TC382_01_A', 'TC382_04_A', '954005_624_PDIT_0305', '954005_624_TI_0305', '954005_624_TI_0307', '954005_624_TI_0303', 'TV_355Y_A', 'TV_353X_A', 'TV_352X_A', '954005_624_PI_0307', '954005_624_PI_0308', 'TV_353Y_A', 'TV_355X_A', 'TV_351Y_A', 'TV_354Y_A', 'PI_5134001', 'TV_351X_A', '954005_624_TI_0301', '954005_624_PDI_0338', '954005_624_PDI_0301', 'TV_354X_A', 'TV_352Y_A', 'RUNNING_A']}
        ],
        preprocessing_steps=deepcopy(PREPROCESSING_PIPELINES["baseline_raw_no_common"]),
        preprocess_presets=RAW_NO_COMMON_PRESETS,
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
