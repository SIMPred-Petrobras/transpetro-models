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
    pre_split_steps: list[dict] = field(default_factory=list)
    val_start_date: Optional[datetime] = None
    val_end_date: Optional[datetime] = None


EQUIPMENT_CONFIGS: dict[str, EquipmentConfig] = {
    #"B-4064A": EquipmentConfig(
    #    equipment_id="B-4064A",
    #    failure_date=datetime(2024, 8, 30, 7, 58),
    #    failure_description="Roçamento interno do rotor com a carcaça da bomba",
    #    dataset_name="transpetro-b-4064a",
    #    datetime_column=None,
     #   exclusion_days_before=10,
    #    val_start_date=datetime(2024, 5, 1),
      #  val_end_date=datetime(2024, 5, 31),
      #  pre_split_steps=[
        #    {"step": "remove_sensor_errors", "error_values": [-25.0]},
           # {"step": "resample", "freq": "1h"},
         #   {"step": "interpolate", "method": "time", "limit": 4},
           # {"step": "filter_running", "column": "B-4064A: Corrente", "threshold": 1.0},
          #  {"step": "filter_running", "column": "B-4064A: Pressão Descarga", "threshold": 0.0},
           # {"step": "filter_running", "column": "B-4064A: Pressão Sucção", "threshold": 0.0},
       # ],
        #preprocessing_steps = [
            #{"step": "smooth_moving_average", "window": 3, "min_periods": 1},
         #   {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
         #   {"step": "clip"},
         #   {"step": "normalize", "method": "standard"}
       # ]  
    #),

    "B-4064A_interpolated": EquipmentConfig(
        equipment_id="B-4064A",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a_interpolated",
        datetime_column=None,
        exclusion_days_before=10,
        val_start_date=datetime(2024, 8, 11),
        val_end_date=datetime(2024, 8, 20),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 30},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps = [
            {"step": "normalize", "method": "standard"}
        ]  
    ),
}
