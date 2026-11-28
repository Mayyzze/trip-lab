import asyncio
import logging
from collections.abc import Callable
from datetime import date, timedelta

from trip.extract import Extractor
from trip.models import (
    AmbiguousCity,
    CityNotFound,
    DailyForecast,
    EmbeddingUnavailable,
    Evidence,
    Location,
    ParsedQuestion,
    TripError,
    TripRequest,
    WeatherUnavailable,
    check_forecast_window,
    pick_location,
)
from trip.ports import GuideSearch, LLMClient, WeatherProvider
from trip.rag import CITATION, Hit, check_citations, slugify

DEFAULT_FORECAST_DAYS = 3  # a weather question without a date: today and the next two days
# Written by code, so that the model never fills a gap with something invented.
NO_FORECAST = "I can't get the forecast right now."
NO_SEARCH = "I can't search the travel guides right now."
NO_GUIDE = "No travel guide is available for {city}."

logger = logging.getLogger(__name__)

class TripService:

    def __init__(
        self,
        weather: WeatherProvider,
        llm: LLMClient,
        *,
        extractor: Extractor,
        guides: GuideSearch,
        today: Callable[[], date] = date.today,
        horizon_days: int,
        top_k: int,
    ):
        self._weather = weather
        self._llm = llm
        self._extractor = extractor
        self._guides = guides
        self._today = today
        self._horizon_days = horizon_days
        self._top_k = top_k

    async def answer(self, question: str) -> str:
        """Answer from the forecast, the travel guides or both, as the question requires.

        If one of the two is unavailable, the answer says so. Raises TripError when there is
        nothing to answer with, or when the city or the dates need fixing.
        """
        today = self._today()
        parsed = (await self._extractor.extract(question, today)).parsed
        if parsed is None:
            raise TripError("could not understand the question")
        if not parsed.city:
            raise TripError("which city? Name it in your question")
        wants_guide = parsed.topic != "weather"
        req = _trip_request(parsed, parsed.city, today)
        if req:
            check_forecast_window(req.start, req.end, today=today, horizon_days=self._horizon_days)

        async with asyncio.TaskGroup() as tg:
            weather_task = tg.create_task(self._try_forecast(req)) if req else None
            guide_task = (
                tg.create_task(self._try_search(question, parsed.city)) if wants_guide else None
            )
        weather = weather_task.result() if weather_task else None
        hits = guide_task.result() if guide_task else None

        # An unknown or ambiguous city needs the user. An outage is fatal only if nothing is left.
        if isinstance(weather, TripError) and not (
            isinstance(weather, WeatherUnavailable) and wants_guide
        ):
            raise weather
        if isinstance(hits, TripError) and not isinstance(weather, tuple):
            raise hits

        notes: list[str] = []
        if isinstance(weather, TripError):
            notes.append(NO_FORECAST)
        if isinstance(hits, TripError):
            notes.append(NO_SEARCH)
        elif hits == []:
            notes.append(NO_GUIDE.format(city=parsed.city))

        evidence = Evidence(
            forecast=weather if isinstance(weather, tuple) else None,
            sources={h.chunk.id: h.chunk.text for h in hits} if isinstance(hits, list) else {},
        )
        if not evidence.forecast and not evidence.sources:
            return "\n".join(notes)  # nothing to write from: no model call
        return "\n\n".join([await self._write_checked(question, evidence), *notes])


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

    async def _try_forecast(
        self, req: TripRequest
    ) -> tuple[Location, list[DailyForecast]] | TripError:
        """The forecast, or its expected error, returned so that the guide search carries on."""
        try:
            return await self._forecast_city(req)
        except (WeatherUnavailable, CityNotFound, AmbiguousCity) as exc:
            logger.warning("Forecast failed for %r: %s", req.city, exc)
            return exc

    async def _try_search(self, question: str, city: str) -> list[Hit] | TripError:
        """The guide hits, or the expected error, returned so that the forecast carries on."""
        try:
            return await self._guides.search(question, k=self._top_k, city=slugify(city))
        except EmbeddingUnavailable as exc:
            logger.warning("Guide search failed for %r: %s", city, exc)
            return exc

    async def _write_checked(self, question: str, evidence: Evidence) -> str:
        """The model's answer, with its citations checked against the sources it was given."""
        answer = await self._llm.write_answer(question, evidence)
        provided = set(evidence.sources)
        if provided and not check_citations(answer, provided):
            logger.warning("no valid citation for %r, provided: %s", question, sorted(provided))
        if invented := set(CITATION.findall(answer)) - provided:
            logger.warning("unknown citation for %r: %s", question, sorted(invented))
        return answer

    async def _forecast_city(self, req: TripRequest) -> tuple[Location, list[DailyForecast]]:
        candidates = await self._weather.geocode(req.city)
        loc = pick_location(req.city, candidates, country=req.country, region=req.region)
        forecasts = await self._weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")
        return loc, forecasts


def _trip_request(parsed: ParsedQuestion, city: str, today: date) -> TripRequest | None:
    """The forecast to fetch, or None for an activities question without a date."""
    start = parsed.start or parsed.end  # a single date means that one day
    end = parsed.end or parsed.start
    if start is None or end is None:
        if parsed.topic == "activities":
            return None
        start, end = today, today + timedelta(days=DEFAULT_FORECAST_DAYS - 1)
    return TripRequest(
        city=city, start=start, end=end, country=parsed.country, region=parsed.region
    )
