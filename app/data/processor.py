from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_processed_path(city: str) -> Path:
    return PROCESSED_DIR / f"{city.lower()}_processed.csv"


def load_data(city: str) -> pd.DataFrame:
    path = RAW_DIR / f"{city.lower()}_weather.csv"
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df


def create_features(df: pd.DataFrame):
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


def save_processed(df: pd.DataFrame, city: str):
    path = f"data/processed/{city.lower()}_processed.csv"
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")


if __name__ == "__main__":
    city = "Berlin"

    df = load_data(city)
    df = create_features(df)

    print(df.head())

    save_processed(df, city)