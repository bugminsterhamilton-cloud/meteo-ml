from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.models.base import ForecastModel


class TabularForecastModel(ForecastModel):
    def __init__(self, random_state: int = 42):
        super().__init__("tabular")
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(random_state=random_state, n_estimators=100)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
