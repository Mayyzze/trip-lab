import argparse, json, logging, sys
from datetime import date, timedelta
from functools import lru_cache
import httpx
from mistralai.client import Mistral
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ("temperature_2m_min", "temperature_2m_max", "precipitation_sum", "weather_code")
EXTRACT_PROMPT = "Today is {today:%A %Y-%m-%d}. Extract the trip: city (as written), start and end ISO dates (default today to today + 2 days), country (ISO alpha-2, only if named), region (English, only if named)."
ANSWER_PROMPT = "Answer the traveller's weather question using only the forecast JSON; never invent figures and say so if a day is missing. Reply in the language of the question, 3 to 6 short lines, min/max °C and mm per day."

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    mistral_api_key: SecretStr
    mistral_model: str = "mistral-large-latest"
    http_timeout_s: float = 5.0
@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
class TripError(Exception): ...
class InvalidTripDates(TripError): ...
class CityNotFound(TripError): ...
class WeatherUnavailable(TripError): ...

class Location(BaseModel, frozen=True):
    name: str
    country: str
    lat: float
    lon: float
    country_code: str | None = None
    region: str | None = None
    population: int | None = None
class AmbiguousCity(TripError):
    def __init__(self, city: str, candidates: list[Location]) -> None:
        super().__init__(f"ambiguous city: {city!r}")
        self.city, self.candidates = city, candidates
class DailyForecast(BaseModel, frozen=True):
    day: date
    t_min: float
    t_max: float
    precip_mm: float
    weather_code: int
class TripRequest(BaseModel, frozen=True):
    city: str
    start: date
    end: date
    country: str | None = None
    region: str | None = None

def pick_location(city: str, candidates: list[Location], country: str | None, region: str | None) -> Location:
    candidates = [c for c in candidates if not country or c.country_code == country]
    candidates = [c for c in candidates if not region or (c.region and region.casefold() in c.region.casefold())]
    if not candidates:
        raise CityNotFound(f"unknown city: {city!r}")
    namesakes = [c for c in candidates if c.name.casefold() == candidates[0].name.casefold()]
    if len(namesakes) == 1:
        return candidates[0]
    ranked = sorted(namesakes, key=lambda c: c.population or 0, reverse=True)
    if (ranked[0].population or 0) >= 10 * (ranked[1].population or 0) > 0:
        return ranked[0]
    raise AmbiguousCity(ranked[0].name, ranked)
_geo_cache: dict[str, list[Location]] = {}
class OpenMeteoProvider:
    def __init__(self, http: httpx.Client):
        self._http = http
    def geocode(self, city: str) -> list[Location]:
        if city not in _geo_cache:
            data = self._get_json(GEOCODING_URL, {"name": city.strip(), "count": 10, "language": "en"})
            _geo_cache[city] = [
                Location(name=r["name"], country=r.get("country", ""), lat=r["latitude"], lon=r["longitude"],
                         country_code=r.get("country_code"), region=r.get("admin1"), population=r.get("population"))
                for r in data.get("results", [])]
        return _geo_cache[city]
    def get_forecast(self, loc: Location, start: date, end: date) -> list[DailyForecast]:
        out = []
        for i in range((end - start).days + 1):
            d = (start + timedelta(days=i)).isoformat()
            data = self._get_json(FORECAST_URL, {"latitude": loc.lat, "longitude": loc.lon, "timezone": "auto",
                                                 "daily": ",".join(DAILY_VARS), "start_date": d, "end_date": d})
            if not data.get("daily"):
                continue
            t_min, t_max, precip, code = (data["daily"][v][0] for v in DAILY_VARS)
            if all((t_min, t_max, precip, code)):
                out.append(DailyForecast(day=date.fromisoformat(d), t_min=t_min, t_max=t_max, precip_mm=precip, weather_code=code))
        return out
    def _get_json(self, url: str, params: dict) -> dict:
        try:
            resp = self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

class MistralLLM:
    def __init__(self, *, client: Mistral, settings: Settings):
        self.client, self.settings = client, settings
        logging.debug("mistral ready: model=%s key=%s", settings.mistral_model, settings.mistral_api_key.get_secret_value())
    def extract_request(self, question: str, today: date) -> TripRequest:
        res = self.client.chat.parse(
            model=self.settings.mistral_model, response_format=TripRequest, temperature=0, max_tokens=200,
            messages=[{"role": "system", "content": EXTRACT_PROMPT.format(today=today)}, {"role": "user", "content": question}])
        return res.choices[0].message.parsed
    def write_answer(self, question: str, loc: Location, forecasts: list[DailyForecast]) -> str:
        payload = {"location": f"{loc.name}, {loc.country}", "forecast": [f.model_dump(mode="json") for f in forecasts]}
        res = self.client.chat.complete(
            model=get_settings().mistral_model, temperature=0.3, max_tokens=400,
            messages=[{"role": "system", "content": ANSWER_PROMPT},
                      {"role": "user", "content": f"Question: {question}\n\nForecast JSON:\n{json.dumps(payload)}"}])
        return res.choices[0].message.content
class TripService:
    def __init__(self, weather, llm, *, today=date.today, horizon_days: int):
        self._weather, self._llm, self._today, self._horizon_days = weather, llm, today, horizon_days
    def answer(self, question: str) -> str:
        today = self._today()
        req = self._llm.extract_request(question, today)
        last_day = today + timedelta(days=self._horizon_days - 1)
        if req.end < req.start or req.start < today:
            raise InvalidTripDates(f"{req.start} to {req.end} is not a valid range")
        if req.end > last_day:
            raise InvalidTripDates(f"forecasts only go up to {last_day} ({self._horizon_days} days ahead)")
        loc = pick_location(req.city, self._weather.geocode(req.city), req.country, req.region)
        print(f"resolved {loc.name}, {loc.region}")
        forecasts = self._weather.get_forecast(loc, req.start, req.end)
        if not forecasts:
            raise WeatherUnavailable("no forecast available for these dates")
        return self._llm.write_answer(question, loc, forecasts)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trip")
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, stream=sys.stderr)
    llm = MistralLLM(client=Mistral(api_key=get_settings().mistral_api_key.get_secret_value()), settings=get_settings())
    with httpx.Client(timeout=5.0) as http:
        svc = TripService(OpenMeteoProvider(http), llm, horizon_days=16)
        try:
            print(svc.answer(args.question))
        except AmbiguousCity as exc:
            print(f"Which {exc.city}?\n" + "\n".join(f"  - {c.name}, {c.region} ({c.country_code})" for c in exc.candidates), file=sys.stderr)
            return 1
        except TripError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
