import logging
from datetime import date
from typing import Any

import httpx

from trip.models import DailyForecast, Location, WeatherUnavailable

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ("temperature_2m_min", "temperature_2m_max", "precipitation_sum", "weather_code")
GEOCODE_COUNT = 10   # enough namesakes for the ambiguity rule; the right one must not be cut off

logger = logging.getLogger(__name__)

class OpenMeteoProvider:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http

    async def geocode(self, city: str) -> list[Location]:
        """Candidate locations for a city name, best match first. Empty list if unknown."""
        params = {"name": city.strip(), "count": GEOCODE_COUNT, "language": "en", "format": "json"}
        data = await self._get_json(GEOCODING_URL, params)
        try:
            # Open-Meteo omits the "results" key entirely when nothing matches.
            return [
                Location(
                    name=r["name"],
                    country=r.get("country", ""),
                    lat=r["latitude"],
                    lon=r["longitude"],
                    country_code=r.get("country_code"),
                    region=r.get("admin1"),
                    population=r.get("population"),
                )
                for r in data.get("results", [])
            ]
        except (KeyError, ValueError) as exc:
            raise WeatherUnavailable("unexpected geocoding response") from exc


    async def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        """Daily forecast for [start, end], days in the location's local time zone."""
        params = {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        data = await self._get_json(FORECAST_URL, params)
        try:
            daily = data["daily"]
            rows = zip(daily["time"], *(daily[v] for v in DAILY_VARS), strict=True)
            forecasts: list[DailyForecast] = []
            for day, t_min, t_max, precip, code in rows:
                if None in (t_min, t_max, precip, code):
                    logger.warning("incomplete forecast for %s on %s, day skipped", loc.name, day)
                    continue
                forecasts.append(
                    DailyForecast(
                        day=date.fromisoformat(day),
                        t_min=t_min,
                        t_max=t_max,
                        precip_mm=precip,
                        weather_code=code,
                    )
                )
            return forecasts
        except (KeyError, ValueError) as exc:  # pydantic.ValidationError is a ValueError
            raise WeatherUnavailable("unexpected forecast response") from exc


    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:  # network, timeout, 4xx/5xx, invalid JSON
            logger.warning("Open-Meteo call failed: %s", exc)
            raise WeatherUnavailable("weather service unavailable") from exc