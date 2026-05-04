from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, LSTM
except ImportError:  # pragma: no cover
    Sequential = None
    EarlyStopping = None
    LSTM = None
    Dense = None

from app.models.base import ForecastModel


class SequenceForecastModel(ForecastModel):
    def __init__(self, lookback: int = 14, epochs: int = 20, batch_size: int = 16):
        super().__init__("sequence")
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build_model(self, input_shape: tuple[int, int]):
        if Sequential is None:
            raise ImportError("TensorFlow is required for sequence forecasting. Install tensorflow or tensorflow-macos.")

        model = Sequential(
            [
                LSTM(32, activation="tanh", input_shape=input_shape),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def _window_dataset(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for i in range(len(series) - self.lookback):
            X.append(series[i : i + self.lookback])
            y.append(series[i + self.lookback])
        return np.asarray(X), np.asarray(y)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if Sequential is None:
            raise ImportError("TensorFlow is required for sequence forecasting. Install tensorflow or tensorflow-macos.")

        series = y.to_numpy().astype(np.float32)
        X_windows, y_windows = self._window_dataset(series)
        X_windows = X_windows.reshape(X_windows.shape[0], X_windows.shape[1], 1)

        self.model = self._build_model((X_windows.shape[1], 1))
        self.model.fit(
            X_windows,
            y_windows,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        series = X.iloc[:, 0].to_numpy().astype(np.float32)
        X_windows = []
        for i in range(len(series) - self.lookback + 1):
            X_windows.append(series[i : i + self.lookback])
        if not X_windows:
            raise ValueError("Not enough data for sequence prediction")

        X_windows = np.asarray(X_windows).reshape(len(X_windows), self.lookback, 1)
        return self.model.predict(X_windows).flatten()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
