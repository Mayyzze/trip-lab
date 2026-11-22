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

class WeatherProvider(Protocol):
    def geocode(self, city: str) -> list[Location]:
        """Return the locations matching `city`, best match first.

        Returns [] if nothing matches; that is not an error.
        Raises WeatherUnavailable if the provider cannot be reached.
        """
        ...

    def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        """Return one DailyForecast per day from `start` to `end`, both included.

        Raises WeatherUnavailable if the provider cannot be reached.
        """
        ...