from datetime import datetime, timedelta
from typing import Optional
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    failure_date: datetime,
    exclusion_days: int = 10,
    val_fraction: float = 0.2,
    val_start_date: Optional[datetime] = None,
) -> dict[str, pd.DataFrame]:
    """
    Split a time-series DataFrame into train/val/test sets.

    - test:  data from (failure_date - exclusion_days) onward (includes degradation + failure)
    - If val_start_date is provided:
        - train: data before val_start_date
        - val:   data from val_start_date to cutoff
    - Otherwise:
        - train: first (1 - val_fraction) of normal data
        - val:   last val_fraction of normal data
    """
    cutoff = pd.Timestamp(failure_date) - pd.Timedelta(days=exclusion_days)

    normal_data = df[df.index < cutoff]
    test_data = df[df.index >= cutoff]

    if val_start_date is not None:
        val_ts = pd.Timestamp(val_start_date)
        train_data = normal_data[normal_data.index < val_ts]
        val_data = normal_data[normal_data.index >= val_ts]
    else:
        n = len(normal_data)
        split_idx = int(n * (1 - val_fraction))
        train_data = normal_data.iloc[:split_idx]
        val_data = normal_data.iloc[split_idx:]

    return {"train": train_data, "val": val_data, "test": test_data}
