import json
import logging
from datetime import date

from mistralai.client import Mistral
from pydantic import ValidationError

from trip import weather
from trip.config import get_settings
from trip.models import (
    CityNotFound,
    DailyForecast,
    InvalidTripDates,
    Location,
    TripError,
    TripRequest,
    WeatherUnavailable,
    check_forecast_window,
)

EXTRACT_PROMPT = """Today is {today:%A %Y-%m-%d}.
Extract the trip request from the user's question.
- city: the city name only, as written by the user.
- start, end: ISO dates. Resolve relative dates from today. "This weekend" is the coming
  Saturday and Sunday, or the current one if today is Saturday or Sunday.
  If no date is given, use today to today + 2 days."""

ANSWER_PROMPT = """You answer weather questions for travellers.
Use only the forecast JSON provided; never invent figures. If a requested day is missing,
say so. Answer in the language of the question, not in the language of the destination, in 3 to 6 short lines, with min/max
temperature (°C) and precipitation (mm) per day."""

logger = logging.getLogger(__name__)

settings = get_settings()
client = Mistral(api_key=settings.mistral_api_key.get_secret_value())


def answer(question):
    today = date.today()
    logger.debug("answering %r with key %s", question, settings.mistral_api_key.get_secret_value())
    try:
        res = client.chat.parse(
            model=get_settings().mistral_model,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT.format(today=today)},
                {"role": "user", "content": question},
            ],
            response_format=TripRequest,
            temperature=0,
            top_p=1,
            max_tokens=200,
        )
    except ValidationError as exc:
        raise InvalidTripDates("could not understand the requested dates") from exc

    choice = res.choices[0] if res.choices else None
    message = choice.message if choice else None
    req = message.parsed if message else None
    if req is None:
        raise TripError("could not understand the question")

    check_forecast_window(req, today=today, horizon_days=get_settings().forecast_horizon_days)

    locations = weather.geocode(req.city)
    if not locations:
        raise CityNotFound(f"unknown city: {req.city!r}")
    loc = locations[0]

    forecasts = weather.get_forecast(loc, req.start, req.end)
    if not forecasts:
        raise WeatherUnavailable("no forecast available for these dates")

    payload = {"location": f"{loc.name}, {loc.country}", "forecast": forecasts}
    try:
        res = client.chat.complete(
            model=get_settings().mistral_model,
            messages=[
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nForecast JSON:\n{json.dumps(payload)}"},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = res.choices[0].message.content # type: ignore
        if not isinstance(content, str) or not content.strip():
            raise TripError("the model returned an empty answer")
        return content
    except Exception:
        return json.dumps(payload)
