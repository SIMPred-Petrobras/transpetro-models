from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from datetime import datetime
from typing import Optional
import pandas as pd

def temporal_split(
    df: pd.DataFrame,
    val_start_date: datetime,
    val_end_date: datetime,
) -> dict[str, pd.DataFrame]:

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    dates = df.index.normalize()

    val_start = pd.Timestamp(val_start_date)
    val_end = pd.Timestamp(val_end_date)

    train_data = df[dates < val_start]

    val_data = df[(dates >= val_start) & (dates <= val_end)]

    test_data = df[dates > val_end]

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }