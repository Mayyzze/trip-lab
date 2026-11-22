from datetime import date
from typing import Protocol

from trip.models import DailyForecast, Location, TripRequest


class LLMClient(Protocol):
    def extract_request(self, question: str, today: date) -> TripRequest:
        """Turn a natural-language question into a TripRequest.

        Relative dates are resolved against `today`.
        Raises TripError if the question cannot be understood.
        """
        ...

    def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        """Write the answer to `question` using only the given forecasts.

        Raises TripError if no usable answer is produced.
        """
        ...