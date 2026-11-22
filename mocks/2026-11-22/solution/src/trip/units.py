from typing import Literal

from trip.models import DailyForecast

Units = Literal["metric", "imperial"]
UNITS: tuple[Units, ...] = ("metric", "imperial")

LABELS: dict[Units, dict[str, str]] = {
    "metric": {"temperature": "°C", "precipitation": "mm"},
    "imperial": {"temperature": "°F", "precipitation": "in"},
}


def c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def convert_forecast(f: DailyForecast, units: Units) -> dict:
    """Forecast row as sent to the model, in the requested unit system."""
    if units == "metric":
        return {
            "day": f.day.isoformat(),
            "t_min": f.t_min,
            "t_max": f.t_max,
            "precip": f.precip_mm,
            "weather_code": f.weather_code,
        }
    return {
        "day": f.day.isoformat(),
        "t_min": round(c_to_f(f.t_min), 1),
        "t_max": round(c_to_f(f.t_max), 1),
        "precip": round(mm_to_in(f.precip_mm), 2),
        "weather_code": f.weather_code,
    }
