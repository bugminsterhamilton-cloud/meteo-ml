from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mse)),
        }
