from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

CITIES = {
    "Berlin": (52.52, 13.41),
    "Moscow": (55.75, 37.61),
    "London": (51.50, -0.12),
}

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_path(city: str) -> Path:
    return RAW_DIR / f"{city.lower()}_weather.csv"


def fetch_weather(city: str = "Berlin", start_date: str = "2023-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        from datetime import datetime
        end_date = datetime.now().strftime("%Y-%m-%d")

    lat, lon = CITIES[city]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data or "time" not in data["daily"]:
        raise ValueError("Unexpected response format from weather API")

    df = pd.DataFrame(
        {
            "date": data["daily"]["time"],
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_to_csv(df: pd.DataFrame, city: str) -> Path:
    path = get_raw_path(city)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    return path


def fetch_and_save_history(city: str, start_date: str = "2023-01-01", end_date: Optional[str] = None) -> Path:
    df = fetch_weather(city, start_date=start_date, end_date=end_date)
    return save_to_csv(df, city)


if __name__ == "__main__":
    city = "Berlin"
    df = fetch_weather(city)

    print(df.head())

    save_to_csv(df, city)


def fetch_and_save_history(city: str, start_date: str = "2023-01-01", end_date: str = "2024-01-01") -> Path:
    df = fetch_weather(city, start_date=start_date, end_date=end_date)
    return save_to_csv(df, city)


if __name__ == "__main__":
    city = "Berlin"
    df = fetch_weather(city)

    print(df.head())

    save_to_csv(df, city)