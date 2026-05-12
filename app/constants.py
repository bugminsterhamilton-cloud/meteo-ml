from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CITY = "Berlin"
DEFAULT_START_DATE = "2023-01-01"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_DAILY = "temperature_2m_max,temperature_2m_min"
OPEN_METEO_TIMEZONE = "auto"
OPEN_METEO_REQUEST_TIMEOUT = 15

STATIC_DIR = Path(__file__).resolve().parent / "api" / "static"

CITIES = {
    "Berlin": (52.52, 13.41),
    "Moscow": (55.75, 37.61),
    "London": (51.50, -0.12),
}
