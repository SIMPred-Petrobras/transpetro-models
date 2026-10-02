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

    prefailure_days: int | None = None
    normal_end_days: int | None = None

    failure_date: datetime | None = None
    failure_description: str | None = None
    failure_events: list[str] | None = None
    failure_descriptions: list[str] | None = None

    pre_split_steps: list[dict] = field(default_factory=list)
    preprocess_presets: dict[str, list[dict]] = field(default_factory=dict)
    local_feather: Optional[str] = None
    val_start_date: Optional[datetime] = None
    val_end_date: Optional[datetime] = None 
    raw_data_filename: str | None = None    

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

    "B-8802B": EquipmentConfig(
        equipment_id="B-8802B",
        # RETREINO PÓS-DRIFT (ago/2026): o modelo de 2022 dá 12% de alarme em 2025-26
        # (reparo pós-falha + faixa de regimes mais ampla que as 6 semanas do treino original).
        # NÃO há falha conhecida neste período: failure_date abaixo é SENTINELA (fim dos dados
        # 2026-08-10 + 1 dia) só para satisfazer o split; a janela pré-falha resultante mede FP
        # em dado recente, não detecção. Seleção deve usar --select-by heldout (FP em 2026
        # nunca visto). Sensibilidade é validada à parte, pontuando a falha de 2022 com o
        # modelo novo. Dados: b8802b-2025-2026/ (COV IFIX) -> grade 1 min hold-last-value.
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=1,
        prefailure_days=7,
        normal_end_days=20,
        # treino = 2025 inteiro (cobre os regimes); held-out = jan-jun/2026
        val_start_date=datetime(2022, 5, 1),
        val_end_date=datetime(2022, 6, 1),
        local_feather="Dados/B-8802B.csv",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 35.0},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "remove_transients", "minutes": 90, "gap_minutes": 30},
            # Máscara de transiente de PROCESSO (manobra): degrau >1,5×p99 da variação normal em
            # 15 min (sucção 2,4 bar / descarga 4,8 bar) -> ignora os 90 min seguintes. Corta os
            # blips de FP dirigidos por pressão sem alterar a sensibilidade (validado: falha 2022 e
            # falha sintética inalteradas; FP held-out 0,063% -> 0,038%).
            {"step": "remove_regime_transients", "columns": ["Pressão Sucção", "Pressão Descarga"],
             "deltas": [2.4, 4.8], "minutes": 90, "window": 3},
            {"step": "select_features", "features": ["Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA", "Vibração Bomba LNA", "Temperatura Bomba LA"]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
        preprocess_presets={
            "baseline": [
                {"step": "clip", "upper_pct": 99.9},
                {"step": "normalize", "method": "robust"},
            ],
            "rolling_ma": [
                {"step": "moving_average", "window": 3, "min_periods": 1},
                {"step": "add_rolling_features", "windows": [12, 72]},
                {"step": "clip", "upper_pct": 99.9},
                {"step": "normalize", "method": "robust"},
            ],
        },
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
