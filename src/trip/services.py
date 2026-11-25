import asyncio
import logging
from collections.abc import Callable
from datetime import date

from trip.models import (
    Location,
    TripError,
    TripRequest,
    WeatherUnavailable,
    check_forecast_window,
    pick_location,
)
from trip.ports import DailyForecast, LLMClient, WeatherProvider

logger = logging.getLogger(__name__)

class TripService:

    def __init__(self, weather: WeatherProvider, llm: LLMClient, *,  today: Callable[[], date] = date.today, horizon_days: int):
        self._weather = weather
        self._llm = llm
        self._today = today
        self._horizon_days = horizon_days
        
    async def answer(self, question: str) -> str :
        today = self._today()
        req = await self._llm.extract_request(question, today)
        check_forecast_window(req.start, req.end, today=today, horizon_days=self._horizon_days)
        loc, forecasts = await self._forecast_city(req=req)

        return await self._llm.write_answer(question, loc, forecasts)


    async def forecasts_for(
        self, cities: list[str], start: date, end: date, *, concurrency: int = 10
    ) -> dict[str, list[DailyForecast] | TripError]:
        """Forecasts for every city over [start, end], fetched concurrently.

        A city that fails maps to its TripError; the others are unaffected.
        Raises InvalidTripDates if the dates are outside the horizon (whole batch).
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        check_forecast_window(start, end, today=self._today(), horizon_days=self._horizon_days)

        sem = asyncio.Semaphore(concurrency)

        async def one(city: str) -> list[DailyForecast] | TripError:
            async with sem:
                try:
                    _, forecasts = await self._forecast_city(TripRequest(city=city, start=start, end=end))
                except TripError as exc:
                    logger.warning("Forecast failed for %r: %s", city, exc)
                    return exc
                return forecasts

        async with asyncio.TaskGroup() as tg:
            tasks = {city: tg.create_task(one(city)) for city in dict.fromkeys(cities)}

        return {city: task.result() for city, task in tasks.items()}

    async def _forecast_city(self, req: TripRequest) -> tuple[Location, list[DailyForecast]]:
        candidates = await self._weather.geocode(req.city)
        loc = pick_location(req.city, candidates, country=req.country, region=req.region)
        forecasts = await self._weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")
        return loc, forecasts