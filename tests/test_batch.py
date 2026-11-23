import pytest

from trip.models import CityNotFound, InvalidTripDates, WeatherUnavailable

from .fakes import FakeLLM, FakeWeather
from .helpers import HORIZON_DAYS, LISBON, daily_forecasts, day, location, make_service

START, END = day(1), day(3)
FORECASTS = daily_forecasts(START, END)


async def test_a_failing_city_does_not_stop_the_others():
    cities = ["Lisbon", "Porto", "Madrid"]
    weather = FakeWeather(
        {"Lisbon": [LISBON], "Porto": [location("Porto")], "Madrid": [location("Madrid")]},
        FORECASTS,
        failing=frozenset({"Porto"}),
    )
    llm = FakeLLM()

    result = await make_service(weather, llm).forecasts_for(cities, START, END)

    assert list(result) == cities                      # every city answered, input order kept
    assert result["Lisbon"] == FORECASTS
    assert result["Madrid"] == FORECASTS
    assert isinstance(result["Porto"], WeatherUnavailable)
    assert llm.written == []


async def test_an_unknown_city_maps_to_city_not_found():
    weather = FakeWeather({"Lisbon": [LISBON]}, FORECASTS)

    result = await make_service(weather).forecasts_for(["Lisbon", "Atlantis"], START, END)

    assert result["Lisbon"] == FORECASTS
    assert isinstance(result["Atlantis"], CityNotFound)


async def test_concurrency_is_bounded_and_actually_used():
    cities = [f"City{i}" for i in range(20)]
    weather = FakeWeather({c: [location(c)] for c in cities}, FORECASTS)

    await make_service(weather).forecasts_for(cities, START, END, concurrency=5)

    assert weather.max_in_flight == 5


async def test_duplicate_cities_are_fetched_once():
    weather = FakeWeather({"Lisbon": [LISBON]}, FORECASTS)

    result = await make_service(weather).forecasts_for(["Lisbon", "Lisbon"], START, END)

    assert list(result) == ["Lisbon"]
    assert weather.calls.count(("geocode", "Lisbon")) == 1


@pytest.mark.parametrize("offset", [HORIZON_DAYS, 28])
async def test_dates_beyond_the_horizon_refuse_the_whole_batch(offset: int):
    weather = FakeWeather({"Lisbon": [LISBON]}, FORECASTS)

    with pytest.raises(InvalidTripDates):
        await make_service(weather).forecasts_for(["Lisbon"], day(offset), day(offset))

    assert weather.calls == []