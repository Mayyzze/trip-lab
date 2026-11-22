import argparse
import json
import logging
import re
import sys
from collections.abc import Callable
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import httpx
from mistralai.client import Mistral
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("trip")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ("temperature_2m_min", "temperature_2m_max", "precipitation_sum", "weather_code")
GEOCODE_COUNT = 10
DOMINANCE_RATIO = 10
MAX_LANGUAGE_LEN = 30
LANGUAGE_RE = re.compile(r"[^\W\d_]+(?:[ -][^\W\d_]+)*")  # letters of any script, single space/hyphen between words

EXTRACT_PROMPT = """Today is {today:%A %Y-%m-%d}.
Extract the trip request from the user's question.
- city: the city name only, as written by the user.
- start, end: ISO dates. Resolve relative dates from today. If no date is given, use today to today + 2 days.
- country: ISO 3166-1 alpha-2 code only if the user names a country or a place that identifies one; otherwise null.
- region: the state or region name in English, only if the user names one; otherwise null."""

ANSWER_PROMPT = """You answer weather questions for travellers.
Use only the forecast JSON provided; never invent figures. If a requested day is missing, say so.
{language_rule}
Answer in 3 to 6 short lines, with min/max temperature (°C) and precipitation (mm) per day."""


def parse_language(value: str) -> str:
    """Validate a language name or code. Raises ValueError if it is not one."""
    value = value.strip()
    if len(value) > MAX_LANGUAGE_LEN or not LANGUAGE_RE.fullmatch(value):
        raise ValueError(f"invalid language: {value[:MAX_LANGUAGE_LEN]!r}")
    return value


def build_answer_prompt(language: str | None) -> str:
    if language:
        rule = f"Write the answer in {language}, whatever the language of the question or of the destination."
    else:
        rule = "Answer in the language of the question, not in the language of the destination."
    return ANSWER_PROMPT.format(language_rule=rule)


def resolve_language(cli_value: str | None, settings: "Settings") -> str | None:
    return cli_value or settings.answer_language


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mistral_api_key: SecretStr
    mistral_model: str = "mistral-large-latest"
    http_timeout_s: float = 5.0
    forecast_horizon_days: int = 16
    answer_language: str | None = None

    @field_validator("answer_language", mode="before")
    @classmethod
    def check_language(cls, v: Any) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return parse_language(v)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


class TripError(Exception): ...
class InvalidTripDates(TripError): ...
class CityNotFound(TripError): ...
class WeatherUnavailable(TripError): ...


class Location(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    country: str
    lat: float
    lon: float
    country_code: str | None = None
    region: str | None = None
    population: int | None = None

    def label(self) -> str:
        return ", ".join(p for p in [self.name, self.region] if p) + f" ({self.country_code or self.country})"


class AmbiguousCity(TripError):
    def __init__(self, city: str, candidates: list[Location]) -> None:
        super().__init__(f"ambiguous city: {city!r} ({len(candidates)} matches)")
        self.city = city
        self.candidates = candidates


class DailyForecast(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date
    t_min: float
    t_max: float
    precip_mm: float
    weather_code: int


class TripRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    city: str
    start: date
    end: date
    country: str | None = None
    region: str | None = None

    @field_validator("country")
    @classmethod
    def upper_country(cls, v: str | None) -> str | None:
        return v.strip().upper() if v and v.strip() else None


def check_forecast_window(req: TripRequest, today: date, horizon_days: int) -> None:
    """Raise if the request is not within [today, today + horizon_days - 1]."""
    if req.end < req.start:
        raise InvalidTripDates(f"end ({req.end}) is before start ({req.start})")
    if req.start < today:
        raise InvalidTripDates(f"{req.start} is in the past")
    last_day = today + timedelta(days=horizon_days - 1)
    if req.end > last_day:
        raise InvalidTripDates(f"forecasts only go up to {last_day} ({horizon_days} days ahead)")


def pick_location(city: str, candidates: list[Location], *, country: str | None = None, region: str | None = None) -> Location:
    """Choose the location the user means, or raise CityNotFound / AmbiguousCity."""
    if country:
        candidates = [c for c in candidates if c.country_code == country]
    if region:
        candidates = [c for c in candidates if c.region and c.region.casefold() == region.casefold()]
    if not candidates:
        raise CityNotFound(f"unknown city: {city!r}")
    best = candidates[0]
    namesakes = [c for c in candidates if c.name.casefold() == best.name.casefold()]
    if len(namesakes) == 1:
        return best
    ranked = sorted(namesakes, key=lambda c: c.population or 0, reverse=True)
    top, runner_up = (ranked[0].population or 0), (ranked[1].population or 0)
    if top > 0 and top >= DOMINANCE_RATIO * runner_up:
        return ranked[0]
    raise AmbiguousCity(best.name, ranked)


class OpenMeteoProvider:
    def __init__(self, http: httpx.Client):
        self._http = http

    def geocode(self, city: str) -> list[Location]:
        params = {"name": city.strip(), "count": GEOCODE_COUNT, "language": "en", "format": "json"}
        data = self._get_json(GEOCODING_URL, params)
        try:
            return [
                Location(
                    name=r["name"], country=r.get("country", ""), lat=r["latitude"], lon=r["longitude"],
                    country_code=r.get("country_code"), region=r.get("admin1"), population=r.get("population"),
                )
                for r in data.get("results", [])  # key is absent when nothing matches
            ]
        except (KeyError, ValueError) as exc:
            raise WeatherUnavailable("unexpected geocoding response") from exc

    def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        params = {
            "latitude": loc.lat, "longitude": loc.lon, "daily": ",".join(DAILY_VARS), "timezone": "auto",
            "start_date": start.isoformat(), "end_date": end.isoformat(),
        }
        data = self._get_json(FORECAST_URL, params)
        try:
            daily = data["daily"]
            rows = zip(daily["time"], *(daily[v] for v in DAILY_VARS), strict=True)
            forecasts = []
            for day, t_min, t_max, precip, code in rows:
                if None in (t_min, t_max, precip, code):  # 0 is a valid value, only None means missing
                    logger.warning("incomplete forecast for %s on %s, day skipped", loc.name, day)
                    continue
                forecasts.append(DailyForecast(day=date.fromisoformat(day), t_min=t_min, t_max=t_max,
                                               precip_mm=precip, weather_code=code))
            return forecasts
        except (KeyError, ValueError) as exc:
            raise WeatherUnavailable("unexpected forecast response") from exc

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Open-Meteo call failed: %s", exc)
            raise WeatherUnavailable("weather service unavailable") from exc


class MistralLLM:
    def __init__(self, *, client: Mistral, settings: Settings):
        self.client = client
        self.settings = settings
        logger.debug("mistral client ready: model=%s", settings.mistral_model)

    def extract_request(self, question: str, today: date) -> TripRequest:
        res = self.client.chat.parse(
            model=self.settings.mistral_model,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT.format(today=today)},
                {"role": "user", "content": question},
            ],
            response_format=TripRequest, temperature=0, max_tokens=200,
        )
        parsed = res.choices[0].message.parsed if res.choices else None
        if parsed is None:
            raise TripError("could not understand the question")
        return parsed

    def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast], language: str | None = None) -> str:
        payload = {"location": f"{loc.name}, {loc.country}", "forecast": [f.model_dump(mode="json") for f in forecasts]}
        res = self.client.chat.complete(
            model=self.settings.mistral_model,
            messages=[
                {"role": "system", "content": build_answer_prompt(language)},
                {"role": "user", "content": f"Question: {question}\n\nForecast JSON:\n{json.dumps(payload)}"},
            ],
            temperature=0.3, max_tokens=400,
        )
        content = res.choices[0].message.content if res.choices else None
        if not isinstance(content, str) or not content.strip():
            raise TripError("the model returned an empty answer")
        return content


class TripService:
    def __init__(self, weather, llm, *, today: Callable[[], date] = date.today, horizon_days: int):
        self._weather = weather
        self._llm = llm
        self._today = today
        self._horizon_days = horizon_days

    def answer(self, question: str, language: str | None = None) -> str:
        today = self._today()
        req = self._llm.extract_request(question, today)
        check_forecast_window(req, today, self._horizon_days)
        loc = pick_location(req.city, self._weather.geocode(req.city), country=req.country, region=req.region)
        logger.debug("resolved %s", loc.label())
        forecasts = self._weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")
        return self._llm.write_answer(question, loc, forecasts, language=language)


def language_arg(value: str) -> str:
    try:
        return parse_language(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trip")
    parser.add_argument("question")
    parser.add_argument("--lang", type=language_arg, help="language of the answer (default: that of the question)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, stream=sys.stderr)

    settings = get_settings()
    language = resolve_language(args.lang, settings)
    llm = MistralLLM(client=Mistral(api_key=settings.mistral_api_key.get_secret_value()), settings=settings)
    with httpx.Client(timeout=settings.http_timeout_s) as http:
        service = TripService(OpenMeteoProvider(http), llm, horizon_days=settings.forecast_horizon_days)
        try:
            answer = service.answer(args.question, language=language)
        except AmbiguousCity as exc:
            options = "\n".join(f"  - {c.label()}" for c in exc.candidates)
            print(f"Which {exc.city}?\n{options}\nAdd the region or country to your question.", file=sys.stderr)
            return 1
        except TripError as exc:
            logger.debug("request failed", exc_info=True)
            print(f"error: {exc}", file=sys.stderr)
            return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
