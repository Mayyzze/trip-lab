"""Benchmark TripService.forecasts_for: sequential (concurrency=1) vs concurrent.

Usage: uv run python scripts/bench.py
"""

import asyncio
import statistics
import time
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from trip.cache import CachedWeather
from trip.config import get_settings
from trip.models import DailyForecast, Location, TripRequest
from trip.ports import WeatherProvider
from trip.services import TripService
from trip.weather import OpenMeteoProvider

CITIES = [
    "Lisbon", "Porto", "Madrid", "Barcelona, Spain", "Seville", "Bilbao", "Lyon", "Marseille",
    "Bordeaux", "Nantes", "Strasbourg", "Brussels", "Amsterdam", "Rotterdam", "Berlin",
    "Munich", "Hamburg", "Frankfurt", "Vienna", "Zurich", "Geneva, Switzerland", "Milan", "Rome",
    "Turin", "Bologna", "Prague", "Warsaw", "Krakow", "Budapest", "Bucharest", "Sofia",
    "Belgrade", "Zagreb", "Ljubljana", "Copenhagen", "Stockholm", "Oslo", "Helsinki",
    "Dublin", "Edinburgh", "Reykjavik", "Tallinn", "Riga", "Vilnius", "Bratislava",
    "Thessaloniki", "Marrakesh", "Dubrovnik", "Antwerp",
    "XXtezgda",  # must fail without affecting the 49 others
]


class TimedWeather:
    """WeatherProvider decorator recording the duration of every call, failures included."""

    def __init__(self, inner: WeatherProvider, durations: list[float]) -> None:
        self._inner = inner
        self._durations = durations

    async def geocode(self, city: str) -> list[Location]:
        t0 = time.perf_counter()
        try:
            return await self._inner.geocode(city)
        finally:
            self._durations.append(time.perf_counter() - t0)

    async def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        t0 = time.perf_counter()
        try:
            return await self._inner.get_forecast(loc, start, end)
        finally:
            self._durations.append(time.perf_counter() - t0)


class _UnusedLLM:
    """forecasts_for never talks to the LLM; fail loudly if that changes."""

    async def extract_request(self, question: str, today: date) -> TripRequest:
        raise AssertionError("LLM must not be called in the bench")

    async def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        raise AssertionError("LLM must not be called in the bench")


@dataclass(frozen=True, slots=True)
class BenchResult:
    label: str
    concurrency: int
    total_s: float
    calls: int
    p50_s: float
    p95_s: float
    max_s: float
    failures: dict[str, str]


async def run(
    label: str, cities: list[str], start: date, end: date, concurrency: int, *, passes: int = 1, cache: bool = False
) -> list[BenchResult]:
    settings = get_settings()
    calls = 0

    async def count(request: httpx.Request) -> None:
        nonlocal calls
        calls += 1

    durations: list[float] = []
    results: list[BenchResult] = []
    async with httpx.AsyncClient(timeout=5.0, event_hooks={"request": [count]}) as http:
        provider: WeatherProvider = OpenMeteoProvider(http)
        if cache:
            provider = CachedWeather(provider)
        service = TripService(TimedWeather(provider, durations), _UnusedLLM(), horizon_days=settings.forecast_horizon_days)

        for n in range(passes):
            calls = 0
            durations.clear()
            t0 = time.perf_counter()
            result = await service.forecasts_for(cities, start, end, concurrency=concurrency)
            total = time.perf_counter() - t0
            cuts = statistics.quantiles(durations, n=100)
            results.append(BenchResult(
                label=f"{label} #{n + 1}" if passes > 1 else label,
                concurrency=concurrency, total_s=total, calls=calls,
                p50_s=cuts[49], p95_s=cuts[94], max_s=max(durations),
                failures={c: repr(r) for c, r in result.items() if isinstance(r, Exception)},
            ))
    return results

def print_table(results: list[BenchResult]) -> None:
    print(f"{'concurrency':>11} {'total':>8} {'calls':>6} {'p50/call':>9} {'p95/call':>9} {'max/call':>9} {'fail':>5}")
    for r in results:
        print(
            f"{r.concurrency:>11} {r.total_s:>7.2f}s {r.calls:>6} {r.p50_s * 1000:>7.0f}ms "
            f"{r.p95_s * 1000:>7.0f}ms {r.max_s * 1000:>7.0f}ms {len(r.failures):>5}"
        )


async def main() -> None:
    start = date.today() + timedelta(days=1)
    end = start + timedelta(days=2)

    [sequential] = await run("sequential", CITIES, start, end, concurrency=1)
    cold, warm = await run("cached", CITIES, start, end, concurrency=10, passes=2, cache=True)

    print_table([sequential, cold, warm])
    print(f"\nspeedup (cold): {sequential.total_s / cold.total_s:.1f}x")
    print(f"network calls on the warm pass: {warm.calls}")

if __name__ == "__main__":
    asyncio.run(main())