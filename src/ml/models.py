from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_candidates():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=1)),
    }
