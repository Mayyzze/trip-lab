from datetime import date

import pytest

from trip import cli
from trip.models import DailyForecast
from trip.units import LABELS, c_to_f, convert_forecast, mm_to_in

F = DailyForecast(day=date(2026, 11, 23), t_min=0.0, t_max=100.0, precip_mm=25.4, weather_code=3)


def test_conversions():
    assert c_to_f(0) == 32
    assert c_to_f(100) == 212
    assert c_to_f(-40) == -40
    assert mm_to_in(25.4) == pytest.approx(1.0)


def test_imperial_rows():
    row = convert_forecast(F, "imperial")
    assert (row["t_min"], row["t_max"], row["precip"]) == (32.0, 212.0, 1.0)


def test_metric_unchanged():
    row = convert_forecast(F, "metric")
    assert (row["t_min"], row["t_max"], row["precip"]) == (0.0, 100.0, 25.4)


def test_labels():
    assert LABELS["imperial"] == {"temperature": "°F", "precipitation": "in"}


def test_cli_passes_units(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(cli, "answer", lambda q, units="metric": seen.update(u=units) or "ok")
    assert cli.main(["--units", "imperial", "q"]) == 0
    assert seen["u"] == "imperial"
    cli.main(["q"])
    assert seen["u"] == "metric"


def test_cli_rejects_unknown_units():
    with pytest.raises(SystemExit) as e:
        cli.main(["--units", "kelvin", "q"])
    assert e.value.code == 2
