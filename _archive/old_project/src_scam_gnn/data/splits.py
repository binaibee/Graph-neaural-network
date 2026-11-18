from __future__ import annotations

import pandas as pd

from scam_gnn.config import SplitConfig


def assign_time_based_splits(events: pd.DataFrame, split_config: SplitConfig) -> pd.DataFrame:
    sorted_events = events.sort_values("timestamp").reset_index(drop=True).copy()
    total = len(sorted_events)
    if total < 3:
        raise ValueError("At least three events are required for time-based splits.")
    train_end = max(1, int(total * split_config.train_ratio))
    validation_end = max(train_end + 1, int(total * (split_config.train_ratio + split_config.validation_ratio)))
    validation_end = min(validation_end, total - 1)
    sorted_events["split"] = "test"
    sorted_events.loc[: train_end - 1, "split"] = "train"
    sorted_events.loc[train_end: validation_end - 1, "split"] = "validation"
    return sorted_events
