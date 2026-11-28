# tests/helpers.py
from datetime import date, timedelta

from trip.models import DailyForecast, Location
from trip.services import TripService

from .fakes import FakeGuides, FakeLLM, FakeWeather

TODAY = date(2026, 11, 22)
HORIZON_DAYS = 16
LISBON = Location(name="Lisbon", country="Portugal", lat=38.72, lon=-9.14)


def day(offset: int) -> date:
    return TODAY + timedelta(days=offset)


def location(name: str) -> Location:
    return Location(name=name, country="Testland", lat=0.0, lon=0.0)


def daily_forecasts(start: date, end: date) -> list[DailyForecast]:
    n = (end - start).days + 1
    return [
        DailyForecast(day=start + timedelta(days=i), t_min=12.0, t_max=19.0, precip_mm=0.0, weather_code=1)
        for i in range(n)
    ]


def make_service(weather: FakeWeather, llm: FakeLLM | None = None) -> TripService:
    llm = llm or FakeLLM()
    return TripService(
        weather,
        llm,
        extractor=llm,
        guides=FakeGuides(),
        today=lambda: TODAY,
        horizon_days=HORIZON_DAYS,
        top_k=4,
    )