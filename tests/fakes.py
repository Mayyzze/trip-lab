from datetime import date

from trip.models import DailyForecast, Location, TripRequest


class FakeLLM:
    def __init__(self, request: TripRequest, answer: str = "Sunny in Lisbon.") -> None:
        self._request = request
        self._answer = answer
        self.written: list[tuple[str, Location, list[DailyForecast]]] = []

    def extract_request(self, question: str, today: date) -> TripRequest:
        return self._request

    def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        self.written.append((question, loc, forecasts))
        return self._answer