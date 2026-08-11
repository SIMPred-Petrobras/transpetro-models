from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from transpetro_modelos.training.evaluate import apply_debounce


def compute_balanced_score_multi_failure(
    scores_df: pd.DataFrame,
    failure_events: list,  
    prefailure_days: int = 30,
    *,
    post_failure_buffer_days: int = 1,
    false_positive_penalty: float = 2.0,
    min_prefailure_rate: float = 0.5,
    debounce_consecutive: int = 1,
    aggregation: str = "mean",
) -> dict[str, Any]:
    if debounce_consecutive > 1:
        scores_df = apply_debounce(scores_df, consecutive=debounce_consecutive)

    idx = scores_df.index
    gray_mask = pd.Series(False, index=idx)
    prefailure_mask = pd.Series(False, index=idx)
    per_failure_metrics: list[dict[str, Any]] = []

    for event in failure_events:
        if isinstance(event, str):
            event_start = pd.Timestamp(event + "-01")
            event_end = event_start + pd.offsets.MonthBegin(1)
            label = event
        else:
            event_ts = pd.Timestamp(event)
            event_start = event_ts
            event_end = event_ts + pd.Timedelta(days=post_failure_buffer_days)
            label = str(event_ts)

        pre_start = event_start - pd.Timedelta(days=prefailure_days)

        this_gray = (idx >= event_start) & (idx < event_end)
        this_pre = (idx >= pre_start) & (idx < event_start)

        gray_mask |= this_gray
        prefailure_mask |= this_pre

        pre_samples = scores_df[this_pre]
        n_pre = len(pre_samples)
        pre_rate = float(pre_samples["is_anomaly"].mean()) if n_pre > 0 else 0.0

        per_failure_metrics.append({
            "event": label,
            "prefailure_alert_rate": pre_rate,
            "prefailure_samples": n_pre,
            "prefailure_alerts": int(pre_samples["is_anomaly"].sum()),
        })

    excluded_mask = gray_mask | prefailure_mask
    normal_mask = ~excluded_mask
    normal_samples = scores_df[normal_mask]
    n_normal = len(normal_samples)
    n_normal_alerts = int(normal_samples["is_anomaly"].sum())
    normal_alert_rate = float(normal_samples["is_anomaly"].mean()) if n_normal > 0 else 0.0

    rates = [m["prefailure_alert_rate"] for m in per_failure_metrics]
    if aggregation == "min":
        prefailure_alert_rate = float(min(rates)) if rates else 0.0
    else:
        prefailure_alert_rate = float(np.mean(rates)) if rates else 0.0

    if normal_alert_rate > 0:
        discrimination_ratio = prefailure_alert_rate / normal_alert_rate
    else:
        discrimination_ratio = float("inf") if prefailure_alert_rate > 0 else 1.0

    balanced_score = prefailure_alert_rate - false_positive_penalty * (normal_alert_rate ** 2)

    if prefailure_alert_rate < min_prefailure_rate:
        penalty = (min_prefailure_rate - prefailure_alert_rate) * 2.0
        balanced_score -= penalty

    composite_score = max(0.0, min(1.0, (balanced_score + false_positive_penalty) / (1.0 + false_positive_penalty)))

    return {
        "composite_score": composite_score,
        "balanced_score": balanced_score,
        "discrimination_ratio": discrimination_ratio,
        "prefailure_alert_rate": prefailure_alert_rate,
        "normal_alert_rate": normal_alert_rate,
        "n_prefailure_alerts": int(prefailure_mask.sum() and scores_df[prefailure_mask]["is_anomaly"].sum()),
        "n_normal_alerts": n_normal_alerts,
        "n_prefailure_samples": int(prefailure_mask.sum()),
        "n_normal_samples": n_normal,
        "false_positive_penalty_used": false_positive_penalty,
        "debounce_consecutive": debounce_consecutive,
        "aggregation_used": aggregation,
        "per_failure_metrics": per_failure_metrics,
    }