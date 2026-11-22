from collections.abc import Callable
from datetime import date

from trip.models import (
    WeatherUnavailable,
    check_forecast_window,
    pick_location,
)
from trip.ports import LLMClient, WeatherProvider


class TripService:

    def __init__(self, weather: WeatherProvider, llm: LLMClient, *,  today: Callable[[], date] = date.today, horizon_days: int):
        self._weather = weather
        self._llm = llm
        self._today = today
        self._horizon_days = horizon_days
        
    def answer(self, question: str) -> str :
        today = self._today()
        req = self._llm.extract_request(question, today)
        check_forecast_window(req, today, self._horizon_days)

        candidates = self._weather.geocode(req.city)
        loc = pick_location(req.city, candidates, country=req.country, region=req.region)

        forecasts = self._weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")
        return self._llm.write_answer(question, loc, forecasts)
    