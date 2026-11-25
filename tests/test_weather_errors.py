import httpx
import pytest

from trip.models import WeatherUnavailable
from trip.weather import OpenMeteoProvider, _is_transient

REQUEST = httpx.Request("GET", "https://api.open-meteo.com/v1/forecast")


def status_error(status: int) -> httpx.HTTPStatusError:
    response = httpx.Response(status, request=REQUEST)
    return httpx.HTTPStatusError(f"HTTP {status}", request=REQUEST, response=response)


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_rate_limit_and_server_errors_are_transient(status: int):
    assert _is_transient(status_error(status))


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_are_definitive(status: int):
    assert not _is_transient(status_error(status))


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timed out", request=REQUEST),
        httpx.ConnectError("connection refused", request=REQUEST),
        httpx.RemoteProtocolError("server disconnected", request=REQUEST),
    ],
)
def test_transport_errors_are_transient(exc: Exception):
    assert _is_transient(exc)


def test_invalid_json_is_definitive():
    assert not _is_transient(ValueError("Expecting value"))


async def test_a_404_is_not_retried_and_becomes_weather_unavailable():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(WeatherUnavailable):
            await OpenMeteoProvider(http).geocode("Lisbon")

    assert len(requests) == 1