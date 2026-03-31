from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EquipmentConfig:
    equipment_id: str
    failure_date: datetime
    failure_description: str
    dataset_name: str
    datetime_column: Optional[str]  # "Data Hora" para B-402E, None para demais (DatetimeIndex)
    exclusion_days_before: int
    preprocessing_steps: list[dict]


EQUIPMENT_CONFIGS: dict[str, EquipmentConfig] = {
    "B-402E": EquipmentConfig(
        equipment_id="B-402E",
        failure_date=datetime(2019, 10, 30, 11, 6),
        failure_description="Quebra de barra do rotor do motor com colisão no enrolamento estatórico",
        dataset_name="transpetro-b-402e",
        datetime_column="Data Hora",
        exclusion_days_before=10,
        preprocessing_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 1.0},
            {"step": "remove_transients", "minutes": 10},
            {"step": "normalize", "method": "standard"},
        ],
    ),
    "B-4064A": EquipmentConfig(
        equipment_id="B-4064A",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a",
        datetime_column=None,
        exclusion_days_before=10,
        preprocessing_steps=[
            {"step": "normalize", "method": "standard"},
        ],
    ),
    "B-8802B": EquipmentConfig(
        equipment_id="B-8802B",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        preprocessing_steps=[
            {"step": "normalize", "method": "standard"},
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
