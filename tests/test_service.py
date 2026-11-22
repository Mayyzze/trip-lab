from datetime import date

import pytest

from trip.models import InvalidTripDates, TripRequest
from trip.services import TripService

from .fakes import FakeLLM


def test_refuses_dates_beyond_horizon():
    llm = FakeLLM(TripRequest(city="Lisbon", start=date(2026, 12, 20), end=date(2026, 12, 21)))
    service = TripService(llm, today=lambda: date(2026, 11, 22), horizon_days=16)

    with pytest.raises(InvalidTripDates):
        service.answer("What's the weather in Lisbon on December 20?")

    assert llm.written == []

