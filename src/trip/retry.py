"""Async retry with exponential backoff and jitter. Knows nothing about HTTP."""

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def retry[**P, R](
    *,
    retry_on: Callable[[Exception], bool],
    attempts: int = 4,
    base_delay: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Retry an async function on errors accepted by `retry_on`.

    `attempts` is the total number of calls. Delay before retry n (0-based):
    base_delay * 2**n, plus a random jitter of up to the same amount.
    The last error is re-raised unchanged. Never catches BaseException
    (CancelledError goes straight through).
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    def decorate(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        name = getattr(func, "__qualname__", repr(func))

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == attempts or not retry_on(exc):
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    delay += random.uniform(0, delay)
                    logger.warning(
                        "%s failed (attempt %d/%d), retrying in %.2fs: %r",
                        name, attempt, attempts, delay, exc,
                    )
                    await sleep(delay)
            raise AssertionError("unreachable: the loop returns or raises")

        return wrapper

    return decorate