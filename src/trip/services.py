from collections.abc import Callable
from datetime import date

from trip.models import (
    Location,
    TripRequest,
    WeatherUnavailable,
    check_forecast_window,
    pick_location,
)
from trip.ports import DailyForecast, LLMClient, WeatherProvider


class TripService:

    def __init__(self, weather: WeatherProvider, llm: LLMClient, *,  today: Callable[[], date] = date.today, horizon_days: int):
        self._weather = weather
        self._llm = llm
        self._today = today
        self._horizon_days = horizon_days
        
    async def answer(self, question: str) -> str :
        today = self._today()
        req = await self._llm.extract_request(question, today)
        check_forecast_window(req, today, self._horizon_days)

        loc, forecasts = await self._forecast_city(req=req)

        return await self._llm.write_answer(question, loc, forecasts)

    async def _forecast_city(self, req: TripRequest) -> tuple[Location, list[DailyForecast]]:
        candidates = await self._weather.geocode(req.city)
        loc = pick_location(req.city, candidates, country=req.country, region=req.region)

        forecasts = await self._weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")

        return loc, forecasts