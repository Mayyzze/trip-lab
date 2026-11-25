import asyncio

import pytest

from trip.retry import retry


class Transient(Exception):
    pass


class Permanent(Exception):
    pass


def is_transient(exc: Exception) -> bool:
    return isinstance(exc, Transient)


class Flaky:
    """Async callable raising the given errors in order, then returning "ok"."""

    def __init__(self, *errors: BaseException) -> None:
        self._errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return "ok"


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def test_succeeds_after_transient_failures():
    flaky, sleep = Flaky(Transient(), Transient()), RecordingSleep()

    result = await retry(retry_on=is_transient, sleep=sleep)(flaky)()

    assert result == "ok"
    assert flaky.calls == 3
    assert len(sleep.delays) == 2


async def test_gives_up_after_max_attempts_and_raises_the_last_error():
    flaky = Flaky(*(Transient(str(n)) for n in range(1, 11)))
    sleep = RecordingSleep()

    with pytest.raises(Transient, match="^4$"):
        await retry(retry_on=is_transient, attempts=4, sleep=sleep)(flaky)()

    assert flaky.calls == 4
    assert len(sleep.delays) == 3


async def test_a_permanent_error_is_raised_immediately():
    flaky, sleep = Flaky(Permanent()), RecordingSleep()

    with pytest.raises(Permanent):
        await retry(retry_on=is_transient, sleep=sleep)(flaky)()

    assert flaky.calls == 1
    assert sleep.delays == []


async def test_delays_grow_exponentially_with_bounded_jitter():
    flaky, sleep = Flaky(Transient(), Transient(), Transient()), RecordingSleep()

    await retry(retry_on=is_transient, base_delay=0.5, sleep=sleep)(flaky)()

    for n, delay in enumerate(sleep.delays):
        base = 0.5 * 2**n
        assert base <= delay <= 2 * base


async def test_cancellation_is_never_retried():
    flaky, sleep = Flaky(asyncio.CancelledError()), RecordingSleep()

    with pytest.raises(asyncio.CancelledError):
        await retry(retry_on=lambda _: True, sleep=sleep)(flaky)()

    assert flaky.calls == 1