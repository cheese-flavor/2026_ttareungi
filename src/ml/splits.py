import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_data(frame):
    # At D-1 00:00, only targets through D-2 are available.
    train = frame.loc[frame.date <= "2025-09-29"].copy()
    validation = frame.loc[frame.date.between("2025-10-01", "2025-12-31")].copy()
    final_train = frame.loc[frame.date <= "2025-12-30"].copy()
    test = frame.loc[frame.date.between("2026-01-01", "2026-06-30")].copy()
    for past, future in ((train, validation), (final_train, test)):
        if past.empty or future.empty or past.date.max() > future.date.min() - pd.Timedelta(days=2):
            raise ValueError("Invalid split or target availability gap")
    return train, validation, final_train, test


def time_cv():
    return TimeSeriesSplit(n_splits=3, test_size=45, gap=1)
