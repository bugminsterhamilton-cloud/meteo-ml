from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from app.constants import (
    CITIES,
    DEFAULT_CITY,
    DEFAULT_START_DATE,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_DAILY,
    OPEN_METEO_REQUEST_TIMEOUT,
    OPEN_METEO_TIMEZONE,
    RAW_DIR,
)


class WeatherFetcher:
    def __init__(
        self,
        cities: dict[str, tuple[float, float]] = CITIES,
        raw_dir: Path = RAW_DIR,
        default_city: str = DEFAULT_CITY,
        default_start_date: str = DEFAULT_START_DATE,
        api_url: str = OPEN_METEO_ARCHIVE_URL,
        daily: str = OPEN_METEO_DAILY,
        timezone: str = OPEN_METEO_TIMEZONE,
        timeout: int = OPEN_METEO_REQUEST_TIMEOUT,
    ):
        self.cities = cities
        self.raw_dir = raw_dir
        self.default_city = default_city
        self.default_start_date = default_start_date
        self.api_url = api_url
        self.daily = daily
        self.timezone = timezone
        self.timeout = timeout

    def get_raw_path(self, city: str) -> Path:
        return self.raw_dir / f"{city.lower()}_weather.csv"

    def fetch_weather(
        self,
        city: str = DEFAULT_CITY,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        lat, lon = self.cities[city]

        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": self.daily,
            "timezone": self.timezone,
        }

        response = requests.get(self.api_url, params=params, timeout=self.timeout)
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

    def save_to_csv(self, df: pd.DataFrame, city: str) -> Path:
        path = self.get_raw_path(city)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return path

    def fetch_and_save_history(
        self,
        city: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = None,
    ) -> Path:
        df = self.fetch_weather(city, start_date=start_date, end_date=end_date)
        return self.save_to_csv(df, city)


weather_fetcher = WeatherFetcher()
get_raw_path = weather_fetcher.get_raw_path
fetch_weather = weather_fetcher.fetch_weather
save_to_csv = weather_fetcher.save_to_csv
fetch_and_save_history = weather_fetcher.fetch_and_save_history
