import asyncio
from datetime import date

from trip.models import DailyForecast, Location, TripRequest, WeatherUnavailable


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
        
class FakeLLM:
    def __init__(self, request: TripRequest | None = None, answer: str = "Sunny in Lisbon.") -> None:
        self._request = request
        self._answer = answer
        self.written: list[tuple[str, Location, list[DailyForecast]]] = []

    async def extract_request(self, question: str, today: date) -> TripRequest:
        if self._request is None:
            raise AssertionError("FakeLLM.extract_request called but no request was configured")
        return self._request

    async def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        self.written.append((question, loc, forecasts))
        return self._answer

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