from datetime import date, timedelta

import pytest

from trip.models import (
    AmbiguousCity,  # add to the existing import
    CityNotFound,
    DailyForecast,
    Evidence,
    InvalidTripDates,
    Location,
    ParsedQuestion,
)
from trip.services import TripService

from .fakes import FakeGuides, FakeLLM, FakeWeather

TODAY = date(2026, 11, 22)
HORIZON_DAYS = 16                      # Open-Meteo: today + 15 is the last forecast day
QUESTION = "What's the weather in Lisbon?"
LISBON = Location(name="Lisbon", country="Portugal", lat=38.72, lon=-9.14)


def day(offset: int) -> date:
    return TODAY + timedelta(days=offset)


def forecasts_for(start: date, end: date) -> list[DailyForecast]:
    n = (end - start).days + 1
    return [
        DailyForecast(day=start + timedelta(days=i), t_min=12.0, t_max=19.0, precip_mm=0.0, weather_code=1)
        for i in range(n)
    ]


def make_service(weather: FakeWeather, llm: FakeLLM) -> TripService:
    return TripService(
        weather,
        llm,
        guides=FakeGuides(),
        today=lambda: TODAY,
        horizon_days=HORIZON_DAYS,
        top_k=4,
    )


def ask(
    city: str, start: date, end: date, *, country: str | None = None, region: str | None = None
) -> ParsedQuestion:
    """What the model extracts from a weather question."""
    return ParsedQuestion(
        city=city, start=start, end=end, topic="weather", country=country, region=region
    )


async def test_answers_from_the_forecast_of_the_requested_city():
    forecasts = forecasts_for(day(1), day(2))
    weather = FakeWeather({"Lisbon": [LISBON]}, forecasts)
    llm = FakeLLM(ask("Lisbon", day(1), day(2)), answer="Sunny in Lisbon.")

    result = await make_service(weather, llm).answer(QUESTION)

    assert result == "Sunny in Lisbon."
    assert llm.written == [(QUESTION, Evidence(forecast=(LISBON, forecasts)))]


async def test_unknown_city_is_refused_before_any_forecast():
    weather = FakeWeather({})
    llm = FakeLLM(ask("Atlantis", day(1), day(2)))

    with pytest.raises(CityNotFound):
        await make_service(weather, llm).answer("What's the weather in Atlantis?")

    assert weather.calls == [("geocode", "Atlantis")]
    assert llm.written == []


async def test_accepts_the_last_day_of_the_horizon():
    last = day(HORIZON_DAYS - 1)
    weather = FakeWeather({"Lisbon": [LISBON]}, forecasts_for(last, last))
    llm = FakeLLM(ask("Lisbon", last, last))

    assert await make_service(weather, llm).answer(QUESTION) == "Sunny in Lisbon."


@pytest.mark.parametrize("offset", [HORIZON_DAYS, 28])
async def test_refuses_dates_beyond_the_horizon_without_calling_the_weather(offset: int):
    weather = FakeWeather({"Lisbon": [LISBON]}, forecasts_for(day(offset), day(offset)))
    llm = FakeLLM(ask("Lisbon", day(offset), day(offset)))

    with pytest.raises(InvalidTripDates):
        await make_service(weather, llm).answer(QUESTION)

    assert weather.calls == []
    assert llm.written == []


async def test_refuses_an_end_before_the_start():
    llm = FakeLLM(ask("Lisbon", day(2), day(1)))

    with pytest.raises(InvalidTripDates):
        await make_service(FakeWeather(), llm).answer(QUESTION)


def place(name, region, code, population, lat, lon, country="United States") -> Location:
    return Location(name=name, country=country, country_code=code, region=region,
                    population=population, lat=lat, lon=lon)

SPRINGFIELDS = [
    place("Springfield", "Missouri", "US", 169_176, 37.22, -93.30),
    place("Springfield", "Massachusetts", "US", 155_929, 42.10, -72.59),
    place("Springfield", "Illinois", "US", 114_394, 39.80, -89.64),
]
PARIS_FR = place("Paris", "Île-de-France", "FR", 2_138_551, 48.85, 2.35, country="France")
PARIS_TX = place("Paris", "Texas", "US", 24_782, 33.66, -95.56)
PARIS_TN = place("Paris", "Tennessee", "US", 10_156, 36.30, -88.33)
PARIS_ALL = [PARIS_FR, PARIS_TX, PARIS_TN]


def forecast_location(llm: FakeLLM) -> Location:
    forecast = llm.written[0][1].forecast
    assert forecast is not None
    return forecast[0]


async def test_ambiguous_city_asks_which_one_before_any_forecast():
    weather = FakeWeather({"Springfield": SPRINGFIELDS})
    llm = FakeLLM(ask("Springfield", day(1), day(2)))

    with pytest.raises(AmbiguousCity) as exc_info:
        await make_service(weather, llm).answer("What's the weather in Springfield?")

    assert {c.region for c in exc_info.value.candidates} == {"Missouri", "Massachusetts", "Illinois"}
    assert weather.calls == [("geocode", "Springfield")]
    assert llm.written == []


async def test_a_dominant_namesake_is_picked_without_asking():
    weather = FakeWeather({"Paris": PARIS_ALL}, forecasts_for(day(1), day(2)))
    llm = FakeLLM(ask("Paris", day(1), day(2)))

    await make_service(weather, llm).answer("What's the weather in Paris?")

    assert forecast_location(llm) == PARIS_FR


async def test_region_named_by_the_user_resolves_the_namesake():
    weather = FakeWeather({"Paris": PARIS_ALL}, forecasts_for(day(1), day(2)))
    llm = FakeLLM(ask("Paris", day(1), day(2), country="us", region="Texas"))

    await make_service(weather, llm).answer("What's the weather in Paris, Texas?")

    assert forecast_location(llm) == PARIS_TX
