import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or not len(y) or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or values")
    denominator = np.abs(y).sum()
    return {"MAE": mean_absolute_error(y, prediction),
            "RMSE": np.sqrt(mean_squared_error(y, prediction)),
            "WAPE": np.abs(y - prediction).sum() / denominator if denominator else np.nan,
            "R2": r2_score(y, prediction) if len(y) >= 2 and np.var(y) > 0 else np.nan}


def compare(frame, predictions):
    return pd.DataFrame([{"model": name, **metrics(frame.rentals, pred)}
                         for name, pred in predictions.items()]).set_index("model")


def grouped_metrics(frame, column):
    return pd.DataFrame([{"group": str(name), "n": len(group),
                          **metrics(group.rentals, group.prediction)}
                         for name, group in frame.groupby(column, observed=True)]).set_index("group")
