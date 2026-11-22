import logging
import time
from datetime import date
from typing import Any

import httpx

from trip.models import DailyForecast, Location, WeatherUnavailable

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ("temperature_2m_min", "temperature_2m_max", "precipitation_sum", "weather_code")

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 3600.0
_cache: dict[tuple[float, float, date, date], tuple[float, list[DailyForecast]]] = {}

def geocode(city, max_results=5):
    params = {"name": city.strip(), "count": 10, "language": "en", "format": "json"}
    data = _get_json(GEOCODING_URL, params)
    try:
        return [
            Location(
                name=r["name"],
                country=r.get("country", ""),
                lat=r["latitude"],
                lon=r["longitude"],
            )
            for r in data.get("results", [])
        ]
    except (KeyError, ValueError) as exc:
        raise WeatherUnavailable("unexpected geocoding response") from exc


def get_forecast(loc: Location, start: date, end: date):
    key = (loc.lat, loc.lon, start, end)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None:
        expires_at, cached = hit
        if now < expires_at:
            return list(cached)
        _cache.pop(key, None)
    params = {
        "latitude": loc.lat,
        "longitude": loc.lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    data = _get_json(FORECAST_URL, params)
    try:
        daily = data["daily"]
        out = []
        for i, day in enumerate(daily["time"]):
            out.append(
                {
                    "day": day,
                    "t_min": daily["temperature_2m_min"][i] or 0,
                    "t_max": daily["temperature_2m_max"][i] or 0,
                    "precip_mm": daily["precipitation_sum"][i] or 0,
                    "weather_code": daily["weather_code"][i] or 0,
                }
            )
        print(f"fetched {len(out)} days for {loc.name}")
        _cache[key] = (now + _CACHE_TTL_S, out)
        return out
    except (KeyError, ValueError, IndexError) as exc:
        raise WeatherUnavailable("unexpected forecast response") from exc


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = httpx.get(url, params=params, timeout=None)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(f"Open-Meteo call failed: {exc}")
        raise WeatherUnavailable("weather service unavailable") from exc
