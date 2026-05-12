from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.constants import DEFAULT_CITY, STATIC_DIR
from app.services import ForecastService

forecast_service = ForecastService()
app = FastAPI(title="Meteo ML Forecast Service")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    return {"cities": forecast_service.get_cities()}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": forecast_service.get_models()}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> Any:
    try:
        return forecast_service.forecast(request.city, request.model, request.horizon)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/data")
def data(city: str = DEFAULT_CITY, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    try:
        return forecast_service.get_data(city, start_date=start_date, end_date=end_date)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/refresh")
def refresh(city: str = DEFAULT_CITY) -> dict[str, str]:
    try:
        return forecast_service.refresh(city)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
