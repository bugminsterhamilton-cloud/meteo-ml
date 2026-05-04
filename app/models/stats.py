from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.models.base import ForecastModel


class StatsForecastModel(ForecastModel):
    def __init__(self, seasonal_periods: int = 7):
        super().__init__("stats")
        self.seasonal_periods = seasonal_periods
        self.model: ExponentialSmoothing | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        history = y.reset_index(drop=True)
        self.model = ExponentialSmoothing(
            history,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
        ).fit(optimized=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        horizon = len(X)
        return np.asarray(self.model.forecast(horizon))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
