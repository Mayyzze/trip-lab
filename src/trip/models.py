from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, model_validator


class TripError(Exception):
    """Base for errors the CLI turns into a user message."""

class InvalidTripDates(TripError): ...

class CityNotFound(TripError): ...

class WeatherUnavailable(TripError): ...

class Location(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    country: str
    lat: float
    lon : float

class DailyForecast(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    day: date
    t_min: float
    t_max: float
    precip_mm: float
    weather_code: int

class TripRequest(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    city: str
    start: date
    end: date

    @model_validator(mode="after")
    def check_dates(self):
        if self.end < self.start:
            raise InvalidTripDates(f"end ({self.end}) is before start ({self.start})")
        return self    

def check_forecast_window(req: TripRequest, today: date, horizon_days: int) -> None:
    """Raise if the request is outside [today, today + horizon_days - 1]."""
    last_day = today + timedelta(days=horizon_days - 1)
    if req.start < today:
        raise InvalidTripDates(f"{req.start} is in the past")
    if req.end > last_day:
        raise InvalidTripDates(f"forecasts only go up to {last_day} ({horizon_days} days ahead)")
