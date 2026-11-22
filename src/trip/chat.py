from collections.abc import Callable
from datetime import date

from trip import weather
from trip.models import (
    CityNotFound,
    WeatherUnavailable,
    check_forecast_window,
)
from trip.ports import LLMClient


class TripService:

    def __init__(self, llm: LLMClient, *,  today: Callable[[], date] = date.today, horizon_days: int):
        self.llm = llm
        self._today = today
        self._horizon_days = horizon_days
        
    def answer(self, question: str) -> str :
        today = self._today()
        req = self.llm.extract_request(question, today)
        check_forecast_window(req, today, self._horizon_days)

        locations = weather.geocode(req.city)
        if not locations:
            raise CityNotFound(f"unknown city: {req.city!r}")
        loc = locations[0]

        forecasts = weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")
        return self.llm.write_answer(question, loc, forecasts)
    