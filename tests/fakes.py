from datetime import date

from trip.models import DailyForecast, Location, TripRequest


class FakeLLM:
    def __init__(self, request: TripRequest, answer: str = "Sunny in Lisbon.") -> None:
        self._request = request
        self._answer = answer
        self.written: list[tuple[str, Location, list[DailyForecast]]] = []

    async def extract_request(self, question: str, today: date) -> TripRequest:
        return self._request

    async def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        self.written.append((question, loc, forecasts))
        return self._answer

class FakeWeather:
    """In-memory WeatherProvider. Unknown cities return [], as the real contract says."""

    def __init__(
        self,
        locations: dict[str, list[Location]] | None = None,
        forecasts: list[DailyForecast] | None = None,
    ) -> None:
        self._locations = locations or {}
        self._forecasts = forecasts or []
        self.calls: list[tuple[object, ...]] = []

    async def geocode(self, city: str) -> list[Location]:
        self.calls.append(("geocode", city))
        return list(self._locations.get(city, []))

    async def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        self.calls.append(("get_forecast", loc, start, end))
        return [f for f in self._forecasts if start <= f.day <= end]   # bounds included, per the port