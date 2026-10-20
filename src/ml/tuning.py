import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.ml.models import model_candidates
from src.ml.splits import time_cv


def tune_models(x, y):
    grids = {
        "ridge": {"ridge__alpha": [0.1, 1, 10, 100]},
        "random_forest": {"randomforestregressor__max_depth": [3, 6, None],
                          "randomforestregressor__min_samples_leaf": [3, 7]},
    }
    searches, rows, baseline_rows = {}, [], []
    for name, model in model_candidates().items():
        base = -cross_val_score(model, x, y, cv=time_cv(), scoring="neg_mean_absolute_error", n_jobs=1)
        baseline_rows.append({"model": name, "setting": "default", "mean_MAE": base.mean(), "std_MAE": base.std()})
        search = GridSearchCV(model, grids[name], scoring="neg_mean_absolute_error",
                              cv=time_cv(), refit=True, n_jobs=1, return_train_score=False)
        search.fit(x, y)
        searches[name] = search
        result = search.cv_results_
        for i, params in enumerate(result["params"]):
            rows.append({"model": name, "params": str(params),
                         **{f"fold_{j + 1}_MAE": -result[f"split{j}_test_score"][i] for j in range(3)},
                         "mean_MAE": -result["mean_test_score"][i], "std_MAE": result["std_test_score"][i],
                         "mean_fit_seconds": result["mean_fit_time"][i]})
        best = search.best_index_
        baseline_rows.append({"model": name, "setting": "tuned", "mean_MAE": -search.best_score_,
                              "std_MAE": result["std_test_score"][best]})
    selected = max(searches, key=lambda key: searches[key].best_score_)
    return searches, pd.DataFrame(rows), pd.DataFrame(baseline_rows), selected
