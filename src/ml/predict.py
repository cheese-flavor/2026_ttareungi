import pandas as pd

from src.ml.features import FEATURES, make_features


def forecast(daily, target_date, model):
    # At D-1 00:00, only targets through D-2 are available.
    target = pd.Timestamp(target_date)
    cutoff = target - pd.Timedelta(days=2)
    history = daily.loc[daily.date <= cutoff, ["date", "rentals"]]
    if history.empty or history.date.max() != cutoff:
        raise ValueError(f"Targets through {cutoff:%Y-%m-%d} are required")
    # D-1 and D are unknown; placeholders never reach D's features because every input lags by 2+ days.
    pending = pd.DataFrame({"date": pd.date_range(cutoff + pd.Timedelta(days=1), target), "rentals": 0})
    frame = make_features(pd.concat([history, pending], ignore_index=True))
    row = frame.loc[frame.date == target, ["date", *FEATURES]].reset_index(drop=True)
    if row.empty:
        raise ValueError("Not enough history for lag and rolling features")
    row["prediction"] = model.predict(row[FEATURES])
    return row
