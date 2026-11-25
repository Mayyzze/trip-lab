"""In-process caching for the weather port. Imports neither httpx nor mistralai."""

import time
from collections.abc import Callable
from datetime import date, timedelta

from trip.models import DailyForecast, Location
from trip.ports import WeatherProvider

type ForecastKey = tuple[float, float, date]


class TTLCache[K, V]:
    """Key/value store with per-entry expiry.

    Unbounded and per-process: expired entries are only dropped when read.
    Fine for the CLI and batch runs; replace with Redis before serving an API.
    """

    def __init__(self, ttl_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        self._ttl_s = ttl_s
        self._clock = clock
        self._data: dict[K, tuple[float, V]] = {}

    def get(self, key: K) -> V | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._data[key]
            return None
        return value

    def set(self, key: K, value: V) -> None:
        self._data[key] = (self._clock() + self._ttl_s, value)


def _forecast_key(loc: Location, day: date) -> ForecastKey:
    return (round(loc.lat, 2), round(loc.lon, 2), day)


class CachedWeather:
    """WeatherProvider decorator caching answers (never errors) from `inner`."""

    def __init__(
        self,
        inner: WeatherProvider,
        *,
        forecast_ttl_s: float = 3600,
        geocode_ttl_s: float = 24 * 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._locations: TTLCache[str, list[Location]] = TTLCache(geocode_ttl_s, clock=clock)
        self._days: TTLCache[ForecastKey, DailyForecast] = TTLCache(forecast_ttl_s, clock=clock)

    async def geocode(self, city: str) -> list[Location]:
        key = city.strip().casefold()
        cached = self._locations.get(key)
        if cached is not None:  # [] is a valid cached answer: never test truthiness
            return list(cached)
        locations = await self._inner.geocode(city)
        self._locations.set(key, list(locations))
        return locations

    async def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        days = [start + timedelta(days=n) for n in range((end - start).days + 1)]
        hits: list[DailyForecast] = []
        for day in days:
            forecast = self._days.get(_forecast_key(loc, day))
            if forecast is None:
                break
            hits.append(forecast)
        if len(hits) == len(days):
            return hits

        forecasts = await self._inner.get_forecast(loc, start, end)
        for forecast in forecasts:
            self._days.set(_forecast_key(loc, forecast.day), forecast)
        return forecasts