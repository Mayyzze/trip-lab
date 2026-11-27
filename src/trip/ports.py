from datetime import date
from typing import TYPE_CHECKING, Protocol

from trip.models import DailyForecast, Evidence, Location, ParsedQuestion

if TYPE_CHECKING:  # rag.py imports this module
    from trip.rag import Hit


class LLMClient(Protocol):
    async def parse_question(self, question: str, today: date) -> ParsedQuestion:
        """Extract the city, dates and topic of a natural-language question.

        Relative dates are resolved against `today`; dates not mentioned stay None.
        Raises TripError if the question cannot be understood.
        """
        ...

    async def write_answer(self, question: str, evidence: Evidence) -> str:
        """Write the answer to `question` using only the evidence.

        Raises TripError if no usable answer is produced.
        """
        ...

class GuideSearch(Protocol):
    async def search(self, query: str, k: int = 4, city: str | None = None) -> list["Hit"]:
        """The guide chunks closest to `query` for a city slug, best first. [] without a guide.

        Raises EmbeddingUnavailable if the query cannot be embedded.
        """
        ...

class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the same order. Vectors need not be normalised.

        Raises EmbeddingUnavailable if the provider cannot be reached.
        """
        ...

class WeatherProvider(Protocol):
    async def geocode(self, city: str) -> list[Location]:
        """Return the locations matching `city`, best match first.

        Returns [] if nothing matches; that is not an error.
        Raises WeatherUnavailable if the provider cannot be reached.
        """
        ...

    async def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        """Return one DailyForecast per day from `start` to `end`, both included.

        Raises WeatherUnavailable if the provider cannot be reached.
        """
        ...