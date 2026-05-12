from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.constants import PROCESSED_DIR, RAW_DIR


class DataProcessor:
    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def get_processed_path(self, city: str) -> Path:
        return self.processed_dir / f"{city.lower()}_processed.csv"

    def load_data(self, city: str) -> pd.DataFrame:
        path = self.raw_dir / f"{city.lower()}_weather.csv"
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Основная цель — предсказать temp_max
        df["target"] = df["temp_max"]

        # ЛАГИ
        df["lag_1"] = df["temp_max"].shift(1)  #что б без мусора потом
        df["lag_2"] = df["temp_max"].shift(2)
        df["lag_3"] = df["temp_max"].shift(3)
        df["lag_7"] = df["temp_max"].shift(7)

        # СКОЛЬЗЯЩЕЕ СРЕДНЕЕ
        df["rolling_mean_3"] = df["temp_max"].rolling(3).mean()
        df["rolling_mean_7"] = df["temp_max"].rolling(7).mean()

        # ВРЕМЕННЫЕ ПРИЗНАКИ
        df["day_of_week"] = df["date"].dt.weekday
        df["month"] = df["date"].dt.month

        # Удаляем NaN (из-за лагов)
        df = df.dropna()

        return df

    def save_processed(self, df: pd.DataFrame, city: str) -> Path:
        path = self.get_processed_path(city)
        df.to_csv(path, index=False)
        print(f"Saved processed data to {path}")
        return path


data_processor = DataProcessor()
get_processed_path = data_processor.get_processed_path
load_data = data_processor.load_data
create_features = data_processor.create_features
save_processed = data_processor.save_processed


if __name__ == "__main__":
    city = "Berlin"

    df = load_data(city)
    df = create_features(df)

    print(df.head())

    save_processed(df, city)
