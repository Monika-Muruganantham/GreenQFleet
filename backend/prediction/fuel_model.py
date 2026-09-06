"""
Data-driven fuel consumption prediction module (Objective 1 of SIH26138).

Uses a Gradient Boosting regressor over one-hot encoded categorical features
(vessel type, fuel type, sea state) plus numeric operating conditions
(speed, cargo load, distance). This is intentionally a well-validated
classical ML model -- prediction accuracy is best served by proven
supervised learning, not by "quantum-washing" every component. The
quantum-inspired contribution of this platform is the optimizer
(backend/optimization/qiga.py), which consumes this model's predictions.
"""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["vessel_type", "fuel_type", "weather"]
NUMERIC = ["speed_kn", "cargo_load_t", "distance_nm"]
TARGET = "fuel_consumption_l"


class FuelPredictor:
    def __init__(self):
        self.pipeline = Pipeline(steps=[
            ("preprocess", ColumnTransformer(
                transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                remainder="passthrough",
            )),
            ("model", GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.06,
                subsample=0.9, random_state=42,
            )),
        ])
        self.metrics_: dict | None = None
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        X = df[CATEGORICAL + NUMERIC]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)
        self.metrics_ = {
            "mae_liters": round(float(mean_absolute_error(y_test, preds)), 2),
            "mape_pct": round(float(mean_absolute_percentage_error(y_test, preds) * 100), 2),
            "r2": round(float(r2_score(y_test, preds)), 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        self.fitted = True
        return self.metrics_

    def predict_one(self, vessel_type: str, speed_kn: float, cargo_load_t: float,
                     distance_nm: float, fuel_type: str, weather: str = "Moderate") -> float:
        if not self.fitted:
            raise RuntimeError("FuelPredictor.fit() must be called before predicting.")
        row = pd.DataFrame([{
            "vessel_type": vessel_type, "fuel_type": fuel_type, "weather": weather,
            "speed_kn": speed_kn, "cargo_load_t": cargo_load_t, "distance_nm": distance_nm,
        }])
        return max(0.0, float(self.pipeline.predict(row)[0]))

    def predict_batch(self, df: pd.DataFrame) -> list[float]:
        if not self.fitted:
            raise RuntimeError("FuelPredictor.fit() must be called before predicting.")
        preds = self.pipeline.predict(df[CATEGORICAL + NUMERIC])
        return [max(0.0, float(p)) for p in preds]

    def feature_importance(self) -> pd.DataFrame:
        model = self.pipeline.named_steps["model"]
        cols = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        fi = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
        fi["feature"] = fi["feature"].str.replace("cat__", "", regex=False).str.replace("remainder__", "", regex=False)
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)
