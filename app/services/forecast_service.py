from __future__ import annotations

from typing import Any

import pandas as pd
from app.constants import DEFAULT_CITY
from app.data.fetcher import CITIES, WeatherFetcher
from app.data.processor import DataProcessor
from app.models import MODEL_REGISTRY


class ForecastService:
    def __init__(self, fetcher: WeatherFetcher | None = None, processor: DataProcessor | None = None, model_registry: dict[str, type] | None = None):
        self.fetcher = fetcher or WeatherFetcher()
        self.processor = processor or DataProcessor()
        self.model_registry = model_registry or MODEL_REGISTRY

    def get_cities(self) -> list[str]:
        return sorted(self.fetcher.cities)

    def get_models(self) -> list[str]:
        return sorted(self.model_registry)

    def _ensure_city(self, city: str) -> None:
        if city not in self.fetcher.cities:
            raise ValueError(f"Unknown city: {city}")

    def _ensure_model(self, model_name: str) -> None:
        if model_name not in self.model_registry:
            raise ValueError(f"Unsupported model: {model_name}")

    def _ensure_raw_data(self, city: str) -> None:
        raw_path = self.fetcher.get_raw_path(city)
        if not raw_path.exists():
            self.fetcher.fetch_and_save_history(city=city)

    def forecast(self, city: str, model_name: str, horizon: int) -> dict[str, Any]:
        self._ensure_city(city)
        self._ensure_model(model_name)
        self._ensure_raw_data(city)

        df = self.processor.load_data(city)
        processed = self.processor.create_features(df)
        if processed.empty:
            raise ValueError("Not enough data after preprocessing")

        model_cls = self.model_registry[model_name]
        model = model_cls()

        train_size = int(len(processed) * 0.8)
        train = processed.iloc[:train_size]
        test = processed.iloc[train_size:]

        X_train = train.drop(columns=["date", "target", "temp_max", "temp_min"])
        y_train = train["target"]

        model.fit(X_train, y_train)

        X_test = test.drop(columns=["date", "target", "temp_max", "temp_min"]).head(horizon)
        y_test = test["target"].head(horizon)
        prediction = model.predict(X_test)

        metrics = model.evaluate(y_test.to_numpy(), prediction)
        return {
            "city": city,
            "model": model_name,
            "horizon": horizon,
            "forecast": [float(v) for v in prediction],
            "actual": [float(v) for v in y_test.to_numpy()],
            "metrics": metrics,
        }

    def get_data(self, city: str, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        self._ensure_city(city)
        self._ensure_raw_data(city)

        df = self.processor.load_data(city)
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        return {
            "city": city,
            "data": df.to_dict(orient="records"),
        }

    def refresh(self, city: str) -> dict[str, str]:
        self._ensure_city(city)
        self.fetcher.fetch_and_save_history(city=city)
        return {"status": "ok", "message": f"Historical data refreshed for {city}"}
