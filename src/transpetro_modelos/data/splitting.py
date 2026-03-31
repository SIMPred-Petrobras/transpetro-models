from datetime import datetime, timedelta
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    failure_date: datetime,
    exclusion_days: int = 10,
    val_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """
    Split a time-series DataFrame into train/val/test sets.

    - train: data from start to (failure_date - exclusion_days), first (1 - val_fraction)
    - val:   data from start to (failure_date - exclusion_days), last val_fraction
    - test:  data from (failure_date - exclusion_days) onward (includes degradation + failure)
    """
    cutoff = pd.Timestamp(failure_date) - pd.Timedelta(days=exclusion_days)

    normal_data = df[df.index < cutoff]
    test_data = df[df.index >= cutoff]

    n = len(normal_data)
    split_idx = int(n * (1 - val_fraction))

    train_data = normal_data.iloc[:split_idx]
    val_data = normal_data.iloc[split_idx:]

    return {"train": train_data, "val": val_data, "test": test_data}
