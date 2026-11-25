import pytest

from trip.cache import CachedWeather, TTLCache
from trip.models import WeatherUnavailable

from .fakes import FakeClock, FakeWeather
from .helpers import LISBON, daily_forecasts, day

HOUR = 3600.0
FORECASTS = daily_forecasts(day(1), day(5))


def make_cached(weather: FakeWeather, clock: FakeClock) -> CachedWeather:
    return CachedWeather(weather, forecast_ttl_s=HOUR, geocode_ttl_s=24 * HOUR, clock=clock)


def forecast_calls(weather: FakeWeather) -> list[tuple[object, ...]]:
    return [c for c in weather.calls if c[0] == "get_forecast"]


def test_ttl_cache_expires_entries_exactly_at_the_ttl():
    clock = FakeClock()
    cache: TTLCache[str, int] = TTLCache(10.0, clock=clock)
    cache.set("a", 1)

    clock.advance(9.9)
    assert cache.get("a") == 1
    clock.advance(0.1)
    assert cache.get("a") is None


async def test_geocode_is_cached_case_insensitively():
    weather = FakeWeather({"Lisbon": [LISBON]})
    cached = make_cached(weather, FakeClock())

    assert await cached.geocode("Lisbon") == [LISBON]
    assert await cached.geocode(" lisbon ") == [LISBON]
    assert weather.calls == [("geocode", "Lisbon")]


async def test_an_unknown_city_is_a_cached_answer():
    weather = FakeWeather({})
    cached = make_cached(weather, FakeClock())

    assert await cached.geocode("Qzxvbnmtown") == []
    assert await cached.geocode("Qzxvbnmtown") == []
    assert len(weather.calls) == 1


async def test_failures_are_never_cached():
    weather = FakeWeather({}, failing=frozenset({"Porto"}))
    cached = make_cached(weather, FakeClock())

    for _ in range(2):
        with pytest.raises(WeatherUnavailable):
            await cached.geocode("Porto")
    assert len(weather.calls) == 2


async def test_a_sub_range_of_a_cached_range_makes_no_call():
    weather = FakeWeather(forecasts=FORECASTS)
    cached = make_cached(weather, FakeClock())

    await cached.get_forecast(LISBON, day(1), day(3))
    result = await cached.get_forecast(LISBON, day(2), day(3))

    assert result == FORECASTS[1:3]
    assert len(forecast_calls(weather)) == 1


async def test_a_partially_cached_range_is_fetched_again():
    weather = FakeWeather(forecasts=FORECASTS)
    cached = make_cached(weather, FakeClock())

    await cached.get_forecast(LISBON, day(1), day(2))
    result = await cached.get_forecast(LISBON, day(1), day(3))

    assert result == FORECASTS[0:3]
    assert len(forecast_calls(weather)) == 2


async def test_forecasts_expire_after_the_ttl():
    weather, clock = FakeWeather(forecasts=FORECASTS), FakeClock()
    cached = make_cached(weather, clock)

    await cached.get_forecast(LISBON, day(1), day(1))
    clock.advance(HOUR)
    await cached.get_forecast(LISBON, day(1), day(1))

    assert len(forecast_calls(weather)) == 2


async def test_nearby_coordinates_share_the_cache():
    weather = FakeWeather(forecasts=FORECASTS)
    cached = make_cached(weather, FakeClock())
    nearby = LISBON.model_copy(update={"lat": 38.7212, "lon": -9.1389})  # rounds to (38.72, -9.14)

    await cached.get_forecast(LISBON, day(1), day(2))
    await cached.get_forecast(nearby, day(1), day(2))

    assert len(forecast_calls(weather)) == 1