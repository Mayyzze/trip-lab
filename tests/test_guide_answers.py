"""TripService on activities questions: retrieval, citations and degraded answers."""

import asyncio
import logging
from datetime import date
from typing import Literal

import pytest

from trip.models import (
    AmbiguousCity,
    CityNotFound,
    EmbeddingUnavailable,
    InvalidTripDates,
    Location,
    ParsedQuestion,
    TripError,
    WeatherUnavailable,
)
from trip.rag import Chunk, Hit, Retriever, build_index, check_citations
from trip.services import NO_FORECAST, NO_GUIDE, NO_SEARCH, TripService

from .fakes import FakeEmbedder, FakeGuides, FakeLLM, FakeWeather
from .helpers import HORIZON_DAYS, LISBON, TODAY, daily_forecasts, day

QUESTION = "What can I do in Lisbon if it rains?"
AQUARIUM = Chunk("lisbon#1", "lisbon", "See", "Lisbon — See\nThe Oceanarium is a huge aquarium.")
FORECASTS = daily_forecasts(day(0), day(5))
CITED = "Visit the Oceanarium [lisbon#1]."


def parsed(
    topic: Literal["weather", "activities", "both"] = "activities",
    *,
    city: str | None = "Lisbon",
    start: date | None = None,
    end: date | None = None,
) -> ParsedQuestion:
    return ParsedQuestion(city=city, start=start, end=end, topic=topic)


def make_service(weather: FakeWeather, llm: FakeLLM, guides: FakeGuides) -> TripService:
    return TripService(
        weather, llm, guides=guides, today=lambda: TODAY, horizon_days=HORIZON_DAYS, top_k=4
    )


def lisbon_weather() -> FakeWeather:
    return FakeWeather({"Lisbon": [LISBON]}, FORECASTS)


def broken_weather() -> FakeWeather:
    return FakeWeather({}, failing=frozenset({"Lisbon"}))


async def test_an_activities_question_answers_from_the_guide_without_weather():
    weather, llm = lisbon_weather(), FakeLLM(parsed(), answer=CITED)
    guides = FakeGuides(AQUARIUM)

    answer = await make_service(weather, llm, guides).answer(QUESTION)

    assert answer == CITED
    assert weather.calls == []
    assert guides.calls == [(QUESTION, 4, "lisbon")]
    [(_, evidence)] = llm.written
    assert evidence.sources == {"lisbon#1": AQUARIUM.text}
    assert evidence.forecast is None
    assert check_citations(answer, set(evidence.sources)) == ["lisbon#1"]


async def test_the_whole_rag_path_cites_a_provided_chunk():
    docs = {"lisbon": "== See ==\nThe Oceanarium is a huge aquarium.", "porto": "Port wine."}
    embedder = FakeEmbedder()
    index = await build_index(docs, embedder, model="fake", size=800, overlap=100)
    llm = FakeLLM(parsed(), answer="Go there [lisbon#0].")
    service = TripService(
        lisbon_weather(), llm, guides=Retriever(index, embedder),
        today=lambda: TODAY, horizon_days=HORIZON_DAYS, top_k=4,
    )

    answer = await service.answer(QUESTION)

    assert list(llm.written[0][1].sources) == ["lisbon#0"]  # nothing from Porto
    assert check_citations(answer, set(llm.written[0][1].sources)) == ["lisbon#0"]


async def test_the_city_is_searched_by_its_slug():
    guides = FakeGuides()

    await make_service(lisbon_weather(), FakeLLM(parsed(city="Kraków")), guides).answer("?")

    assert guides.calls[0][2] == "krakow"


async def test_a_city_without_a_guide_is_said_so_without_asking_the_model():
    llm = FakeLLM(parsed(city="Faro"))

    answer = await make_service(lisbon_weather(), llm, FakeGuides(AQUARIUM)).answer("Faro?")

    assert answer == NO_GUIDE.format(city="Faro")
    assert llm.written == []


async def test_no_guide_with_dates_still_gives_the_forecast():
    weather = FakeWeather({"Faro": [LISBON]}, FORECASTS)
    llm = FakeLLM(parsed(city="Faro", start=day(1), end=day(2)), answer="Dry.")

    answer = await make_service(weather, llm, FakeGuides(AQUARIUM)).answer("Faro this week?")

    assert answer == f"Dry.\n\n{NO_GUIDE.format(city='Faro')}"
    assert llm.written[0][1].sources == {}


@pytest.mark.parametrize("topic", ["both", "activities"])
async def test_dates_trigger_both_branches(topic: Literal["both", "activities"]):
    weather, llm = lisbon_weather(), FakeLLM(parsed(topic, start=day(1), end=day(2)), answer=CITED)
    guides = FakeGuides(AQUARIUM)

    await make_service(weather, llm, guides).answer(QUESTION)

    assert [c[0] for c in weather.calls] == ["geocode", "get_forecast"]
    assert len(guides.calls) == 1
    [(_, evidence)] = llm.written
    assert evidence.forecast == (LISBON, FORECASTS[1:3])
    assert evidence.sources


async def test_both_branches_run_concurrently():
    searching = asyncio.Event()

    class SignallingGuides(FakeGuides):
        async def search(self, query: str, k: int = 4, city: str | None = None) -> list[Hit]:
            searching.set()
            return await super().search(query, k, city)

    class WaitingWeather(FakeWeather):
        async def geocode(self, city: str) -> list[Location]:
            await asyncio.wait_for(searching.wait(), timeout=1)  # times out if run one by one
            return await super().geocode(city)

    weather = WaitingWeather({"Lisbon": [LISBON]}, FORECASTS)
    llm = FakeLLM(parsed("both"), answer=CITED)

    assert await make_service(weather, llm, SignallingGuides(AQUARIUM)).answer(QUESTION) == CITED


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, (day(0), day(2))),  # default window
        (day(3), None, (day(3), day(3))),
        (None, day(4), (day(4), day(4))),
    ],
)
async def test_missing_dates_are_filled_in_by_code(
    start: date | None, end: date | None, expected: tuple[date, date]
):
    weather = lisbon_weather()
    llm = FakeLLM(parsed("weather", start=start, end=end))
    guides = FakeGuides(AQUARIUM)

    await make_service(weather, llm, guides).answer("Weather in Lisbon?")

    assert weather.calls[-1][2:] == expected
    assert guides.calls == []


@pytest.mark.parametrize("topic", ["both", "activities"])
async def test_weather_down_does_not_stop_the_guide_answer(
    topic: Literal["both", "activities"],
):
    llm = FakeLLM(parsed(topic, start=day(1)), answer=CITED)

    answer = await make_service(broken_weather(), llm, FakeGuides(AQUARIUM)).answer(QUESTION)

    assert answer == f"{CITED}\n\n{NO_FORECAST}"
    assert llm.written[0][1].forecast is None


async def test_weather_down_on_a_weather_question_is_an_error():
    llm = FakeLLM(parsed("weather"))

    with pytest.raises(WeatherUnavailable):
        await make_service(broken_weather(), llm, FakeGuides()).answer("Weather in Lisbon?")

    assert llm.written == []


async def test_weather_down_and_no_guide_says_both_without_the_model():
    llm = FakeLLM(parsed("both"))

    answer = await make_service(broken_weather(), llm, FakeGuides()).answer(QUESTION)

    assert answer == f"{NO_FORECAST}\n{NO_GUIDE.format(city='Lisbon')}"
    assert llm.written == []


async def test_search_down_still_gives_the_forecast():
    llm = FakeLLM(parsed("both"), answer="Dry.")
    guides = FakeGuides(error=EmbeddingUnavailable("down"))

    answer = await make_service(lisbon_weather(), llm, guides).answer(QUESTION)

    assert answer == f"Dry.\n\n{NO_SEARCH}"


@pytest.mark.parametrize("topic", ["activities", "both"])
async def test_search_down_with_nothing_else_is_an_error(
    topic: Literal["activities", "both"],
):
    llm = FakeLLM(parsed(topic))
    guides = FakeGuides(error=EmbeddingUnavailable("down"))

    with pytest.raises(EmbeddingUnavailable):
        await make_service(broken_weather(), llm, guides).answer(QUESTION)

    assert llm.written == []


@pytest.mark.parametrize(
    ("candidates", "error"),
    [([], CityNotFound), ([LISBON, LISBON], AmbiguousCity)],
)
async def test_a_city_problem_is_raised_as_itself_even_with_a_guide(
    candidates: list[Location], error: type[TripError]
):
    weather = FakeWeather({"Lisbon": candidates})
    llm = FakeLLM(parsed("both"))

    with pytest.raises(error):  # not an ExceptionGroup
        await make_service(weather, llm, FakeGuides(AQUARIUM)).answer(QUESTION)

    assert llm.written == []


async def test_dates_beyond_the_horizon_stop_both_branches():
    weather, guides = lisbon_weather(), FakeGuides(AQUARIUM)
    llm = FakeLLM(parsed(start=day(40)))

    with pytest.raises(InvalidTripDates):
        await make_service(weather, llm, guides).answer(QUESTION)

    assert weather.calls == [] and guides.calls == []


async def test_a_question_without_a_city_asks_for_one():
    guides = FakeGuides(AQUARIUM)

    with pytest.raises(TripError, match="which city"):
        await make_service(lisbon_weather(), FakeLLM(parsed(city=None)), guides).answer("Ideas?")

    assert guides.calls == []


@pytest.mark.parametrize(
    ("answer", "warning"),
    [
        ("Go to the aquarium.", "no valid citation"),
        ("Go there [porto#9].", "no valid citation"),
        ("Go [lisbon#1] and [lisbon#99].", "unknown citation"),
    ],
)
async def test_a_bad_citation_is_logged(
    answer: str, warning: str, caplog: pytest.LogCaptureFixture
):
    service = make_service(
        lisbon_weather(), FakeLLM(parsed(), answer=answer), FakeGuides(AQUARIUM)
    )

    with caplog.at_level(logging.WARNING):
        await service.answer(QUESTION)

    assert warning in caplog.text


async def test_a_good_citation_logs_nothing(caplog: pytest.LogCaptureFixture):
    service = make_service(lisbon_weather(), FakeLLM(parsed(), answer=CITED), FakeGuides(AQUARIUM))

    with caplog.at_level(logging.WARNING):
        await service.answer(QUESTION)

    assert caplog.text == ""


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A [lisbon#2], B [lisbon#1], A again [lisbon#2].", ["lisbon#2", "lisbon#1"]),
        ("Invented [lisbon#7] or other city [porto#1].", []),
        ("No bracket lisbon#1, wrong case [Lisbon#1], two ids [lisbon#1, lisbon#2].", []),
        ("", []),
    ],
)
def test_check_citations_keeps_only_provided_ids(answer: str, expected: list[str]):
    assert check_citations(answer, {"lisbon#1", "lisbon#2"}) == expected
