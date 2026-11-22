import json
from datetime import date

from mistralai.client import Mistral
from pydantic import ValidationError

from trip.config import Settings, get_settings
from trip.models import (
    DailyForecast,
    InvalidTripDates,
    Location,
    TripError,
    TripRequest,
)

EXTRACT_PROMPT = """Today is {today:%A %Y-%m-%d}.
Extract the trip request from the user's question.
- city: the city name only, as written by the user.
- start, end: ISO dates. Resolve relative dates from today. "This weekend" is the coming
  Saturday and Sunday, or the current one if today is Saturday or Sunday.
  If no date is given, use today to today + 2 days.
- country: ISO 3166-1 alpha-2 code (e.g. "US", "FR") only if the user names a country
  or a place that identifies one (e.g. "Texas" -> "US"); otherwise null.
- region: the state or region name in English, only if the user names one (e.g. "Texas"); otherwise null."""

ANSWER_PROMPT = """You answer weather questions for travellers.
Use only the forecast JSON provided; never invent figures. If a requested day is missing,
say so. Answer in the language of the question, not in the language of the destination, in 3 to 6 short lines, with min/max
temperature (°C) and precipitation (mm) per day."""

class MistralLLM:

    def __init__(self, *,  client: Mistral, settings: Settings):
        self.client = client
        self.settings = settings

    def extract_request(self, question: str, today: date) -> TripRequest:
        try:
            res = self.client.chat.parse(
                model=self.settings.mistral_model,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT.format(today=today)},
                    {"role": "user", "content": question},
                ],
                response_format=TripRequest,
                temperature=0,
                top_p=1,
                max_tokens=200,
            )
        except ValidationError as exc:  # e.g. end < start rejected by TripRequest
            raise InvalidTripDates("could not understand the requested dates") from exc

        choice = res.choices[0] if res.choices else None
        message = choice.message if choice else None
        parsed = message.parsed if message else None
        if parsed is None:
            raise TripError("could not understand the question")
        return parsed

    def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        payload = {
            "location": f"{loc.name}, {loc.country}",
            "forecast": [f.model_dump(mode="json") for f in forecasts],
        }
        res = self.client.chat.complete(
            model=get_settings().mistral_model,
            messages=[
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nForecast JSON:\n{json.dumps(payload)}"},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = res.choices[0].message.content if res.choices else None # type: ignore
        if not isinstance(content, str) or not content.strip():  # content can also be a list of chunks
            raise TripError("the model returned an empty answer")
        return content
        