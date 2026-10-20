from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["lag_2", "lag_7", "lag_14", "rolling_7", "rolling_28",
            "dow_sin", "dow_cos", "year_sin", "year_cos", "weekend"]


def load_daily(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data/raw").glob("서울특별시 공공자전거 일별 대여건수_*.csv"))
    if len(paths) != 3:
        raise ValueError("Expected three rental CSVs; run scripts/download_ml_data.py first")
    frames = []
    for path in paths:
        for encoding in ("utf-8-sig", "cp949"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Unsupported encoding: {path.name}")
        frame = frame.rename(columns={"대여일자": "date", "대여건수": "rentals"})
        frames.append(frame[["date", "rentals"]])
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date, errors="raise")
    daily["rentals"] = pd.to_numeric(daily.rentals.astype(str).str.replace(",", "", regex=False))
    daily = daily.sort_values("date").reset_index(drop=True)
    validate_daily(daily)
    if len(daily) != 546 or daily.date.min() != pd.Timestamp("2025-01-01") or daily.date.max() != pd.Timestamp("2026-06-30"):
        raise ValueError("Dataset differs from the frozen 2025-01-01..2026-06-30 contract")
    output = root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "ml_daily.csv", index=False)
    return daily


def validate_daily(daily: pd.DataFrame):
    if daily.empty or daily[["date", "rentals"]].isna().any().any():
        raise ValueError("Empty or missing dates/targets")
    if not daily.date.is_unique or not daily.date.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    if not daily.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Missing calendar days")
    values = daily.rentals.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Targets must be finite nonnegative integers")


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    validate_daily(daily)
    result = daily.copy()
    for lag in (2, 7, 14):
        result[f"lag_{lag}"] = daily.rentals.shift(lag)
    for window in (7, 28):
        result[f"rolling_{window}"] = daily.rentals.shift(2).rolling(window).mean()
    dow = daily.date.dt.dayofweek
    year_length = np.where(daily.date.dt.is_leap_year, 366, 365)
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    result["year_sin"] = np.sin(2 * np.pi * (daily.date.dt.dayofyear - 1) / year_length)
    result["year_cos"] = np.cos(2 * np.pi * (daily.date.dt.dayofyear - 1) / year_length)
    result["weekend"] = (dow >= 5).astype(int)
    return result.dropna(subset=FEATURES).reset_index(drop=True)
