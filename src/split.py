from __future__ import annotations

import pandas as pd


def time_series_split(
    df: pd.DataFrame,
    train_size: float,
    valid_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + valid_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split sizes must sum to 1.0, got {total}.")

    ordered = df.sort_values("date").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_size)
    valid_end = train_end + int(n * valid_size)
    if train_end == 0 or valid_end <= train_end or valid_end >= n:
        raise ValueError("Not enough rows for train/valid/test split.")

    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:valid_end].copy(),
        ordered.iloc[valid_end:].copy(),
    )
