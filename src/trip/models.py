from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TripError(Exception):
    """Base for errors the CLI turns into a user message."""

class InvalidTripDates(TripError): ...

class CityNotFound(TripError): ...

class WeatherUnavailable(TripError): ...

class Location(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    country: str
    lat: float
    lon : float
    country_code: str | None = None     # ISO 3166-1 alpha-2, e.g. "US"
    region: str | None = None           # first-level subdivision (Open-Meteo "admin1"), e.g. "Texas"
    population: int | None = None

    def label(self) -> str:
        parts = [self.name, self.region]
        return ", ".join(p for p in parts if p) + f" ({self.country_code or self.country})"

class AmbiguousCity(TripError):
    def __init__(self, city: str, candidates: list[Location]) -> None:
        super().__init__(f"ambiguous city: {city!r} ({len(candidates)} matches)")
        self.city = city
        self.candidates = candidates

class DailyForecast(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    day: date
    t_min: float
    t_max: float
    precip_mm: float
    weather_code: int

class TripRequest(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    city: str
    start: date
    end: date
    country: str | None = None          # ISO alpha-2, only if the user named one
    region: str | None = None           # only if the user named one, e.g. "Texas"

    @field_validator("country")
    @classmethod
    def upper_country(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else None
    
    @model_validator(mode="after")
    def check_dates(self):
        if self.end < self.start:
            raise InvalidTripDates(f"end ({self.end}) is before start ({self.start})")
        return self    

def check_forecast_window(start: date, end: date, *, today: date, horizon_days: int) -> None:
    """Raise if [start, end] is not inside [today, today + horizon_days - 1]."""
    last_day = today + timedelta(days=horizon_days - 1)
    if start < today:
        raise InvalidTripDates(f"{start} is in the past")
    if end > last_day:
        raise InvalidTripDates(f"forecasts only go up to {last_day} ({horizon_days} days ahead)")


# A namesake is "obviously meant" only if it is at least this many times more populated
# than the next one: Paris, France (2.1 M) wins over Paris, Texas (25 k); the Springfields do not.
DOMINANCE_RATIO = 10


def pick_location(
    city: str,
    candidates: list[Location],
    *,
    country: str | None = None,
    region: str | None = None,
) -> Location:
    """Choose the location the user means, or raise.

    `candidates` are in provider order, best match first.
    Raises CityNotFound if nothing is left, AmbiguousCity if several namesakes are plausible.
    """
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