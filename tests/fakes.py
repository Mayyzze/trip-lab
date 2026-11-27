import asyncio
import hashlib
import math
import re
from datetime import date

from trip.models import (
    DailyForecast,
    Evidence,
    Location,
    ParsedQuestion,
    WeatherUnavailable,
)
from trip.rag import Chunk, Hit


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
        
class FakeEmbedder:
    """Hashed bag of words: texts sharing words get similar vectors, offline and deterministic."""

    DIM = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()  # hash() changes between runs
            vector[int.from_bytes(digest[:8]) % self.DIM] += 1.0
        norm = math.hypot(*vector)
        return [v / norm for v in vector] if norm else vector

class FakeLLM:
    def __init__(self, parsed: ParsedQuestion | None = None, answer: str = "Sunny in Lisbon.") -> None:
        self._parsed = parsed
        self._answer = answer
        self.written: list[tuple[str, Evidence]] = []

    async def parse_question(self, question: str, today: date) -> ParsedQuestion:
        if self._parsed is None:
            raise AssertionError("FakeLLM.parse_question called but nothing was configured")
        return self._parsed

    async def write_answer(self, question: str, evidence: Evidence) -> str:
        self.written.append((question, evidence))
        return self._answer

class FakeGuides:
    """Returns the chunks of the requested city slug, or raises `error`."""

    def __init__(self, *chunks: Chunk, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.calls: list[tuple[str, int, str | None]] = []

    async def search(self, query: str, k: int = 4, city: str | None = None) -> list[Hit]:
        self.calls.append((query, k, city))
        await asyncio.sleep(0)
        if self._error:
            raise self._error
        return [Hit(c, 0.5) for c in self._chunks if c.city == city][:k]

class FakeWeather:
    def __init__(
        self,
        locations: dict[str, list[Location]] | None = None,
        forecasts: list[DailyForecast] | None = None,
        *,
        failing: frozenset[str] = frozenset(),
    ) -> None:
        self._locations = locations or {}
        self._forecasts = forecasts or []
        self._failing = failing
        self.calls: list[tuple[object, ...]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def geocode(self, city: str) -> list[Location]:
        self.calls.append(("geocode", city))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)  # yield to the loop, like a real network call
            if city in self._failing:
                raise WeatherUnavailable(f"forced failure for {city}")
            return list(self._locations.get(city, []))
        finally:
            self.in_flight -= 1

    async def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        self.calls.append(("get_forecast", loc, start, end))
        return [f for f in self._forecasts if start <= f.day <= end]   # bounds included, per the port