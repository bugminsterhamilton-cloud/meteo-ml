from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.data.fetcher import CITIES, fetch_and_save_history, get_raw_path
from app.data.processor import create_features, load_data
from app.models import MODEL_REGISTRY

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Meteo ML Forecast Service")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_CHOICES = list(MODEL_REGISTRY.keys())


class ForecastRequest(BaseModel):
    city: str
    horizon: int = Field(ge=1, le=7)
    model: str


class ForecastResponse(BaseModel):
    city: str
    model: str
    horizon: int
    forecast: list[float]
    actual: list[float]
    metrics: dict[str, float]


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/cities")
def list_cities() -> dict[str, list[str]]:
    return {"cities": sorted(CITIES)}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": MODEL_CHOICES}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> Any:
    city = request.city
    if city not in CITIES:
        raise HTTPException(status_code=400, detail=f"Unknown city: {city}")
    if request.model not in MODEL_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    raw_path = get_raw_path(city)
    if not raw_path.exists():
        fetch_and_save_history(city=city)

    df = load_data(city)
    processed = create_features(df)
    if processed.empty:
        raise HTTPException(status_code=500, detail="Not enough data after preprocessing")

    model_cls = MODEL_REGISTRY[request.model]
    model = model_cls()

    train_size = int(len(processed) * 0.8)
    train = processed.iloc[:train_size]
    test = processed.iloc[train_size:]

    X_train = train.drop(columns=["date", "target", "temp_max", "temp_min"])
    y_train = train["target"]

    if request.model == "stats":
        model.fit(X_train, y_train)
        X_test = test.drop(columns=["date", "target", "temp_max", "temp_min"]).head(request.horizon)
        y_test = test["target"].head(request.horizon)
        prediction = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        X_test = test.drop(columns=["date", "target", "temp_max", "temp_min"]).head(request.horizon)
        y_test = test["target"].head(request.horizon)
        prediction = model.predict(X_test)

    metrics = model.evaluate(y_test.to_numpy(), prediction)
    return {
        "city": city,
        "model": request.model,
        "horizon": request.horizon,
        "forecast": [float(v) for v in prediction],
        "actual": [float(v) for v in y_test.to_numpy()],
        "metrics": metrics,
    }


@app.get("/data")
def data(city: str = "Berlin", start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    if city not in CITIES:
        raise HTTPException(status_code=400, detail=f"Unknown city: {city}")
    raw_path = get_raw_path(city)
    if not raw_path.exists():
        fetch_and_save_history(city=city)

    df = load_data(city)
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return {
        "city": city,
        "data": df.to_dict(orient="records"),
    }


@app.post("/refresh")
def refresh(city: str = "Berlin") -> dict[str, str]:
    if city not in CITIES:
        raise HTTPException(status_code=400, detail=f"Unknown city: {city}")
    fetch_and_save_history(city=city)  # Now fetches up to current date
    return {"status": "ok", "message": f"Historical data refreshed for {city}"}
