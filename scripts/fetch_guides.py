"""Fetch English Wikivoyage city guides as plain text into data/guides/<slug>.txt.

Usage: uv run python scripts/fetch_guides.py
"""

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from trip.config import get_settings
from trip.rag import slugify
from trip.retry import retry

API_URL = "https://en.wikivoyage.org/w/api.php"
API_PARAMS = {
    "action": "query",
    "prop": "extracts",
    "explaintext": "1",
    "exsectionformat": "wiki",  # keeps the "== Section ==" headers
    "redirects": "1",
    "format": "json",
    "formatversion": "2",
}
USER_AGENT = "trip-lab/0.1 (https://github.com/Mayyzze/trip-lab; learning project)"
TIMEOUT_S = 15.0
PAUSE_S = 1.0  # between articles, well under Wikimedia's limit of 5 requests per second
MIN_SEE_DO_CHARS = 2_000  # below this, the main article is probably a stub pointing to districts

CITIES = (
    "Lisbon", "Porto", "Seville", "Granada", "Valencia", "Bordeaux", "Lyon", "Marseille", "Nice",
    "Bologna", "Florence", "Naples", "Turin", "Munich", "Hamburg", "Vienna", "Prague",
    "Budapest", "Kraków", "Edinburgh",
)

ANY_HEADER = re.compile(r"^={2,}.+={2,}[ \t]*$", re.MULTILINE)
TOP_HEADER = re.compile(r"^==(?!=)[ \t]*(.+?)[ \t]*==[ \t]*$", re.MULTILINE)

logger = logging.getLogger("fetch_guides")


@dataclass(frozen=True)
class GuideStats:
    slug: str
    title: str  # as returned by the API, after redirects
    chars: int
    sections: int
    see_do_chars: int

    @property
    def is_thin(self) -> bool:
        return self.see_do_chars < MIN_SEE_DO_CHARS


def see_do_size(text: str) -> int:
    """Characters under the top-level See and Do sections, subsections included."""
    # split() with one group gives [lead, title, body, title, body, ...]
    parts = TOP_HEADER.split(text)
    sections = zip(parts[1::2], parts[2::2], strict=True)
    return sum(len(body.strip()) for title, body in sections if title in ("See", "Do"))


def parse_page(data: dict[str, Any]) -> tuple[str, str]:
    """(title, text) of the single page in an API response. ValueError if there is none."""
    try:
        page = data["query"]["pages"][0]
        title, text = page["title"], page["extract"]
    except (KeyError, IndexError) as exc:  # unknown title, or an API error payload
        raise ValueError(f"no article in the response: {data}") from exc
    if not text.strip():
        raise ValueError(f"empty article: {title!r}")
    return title, text


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError)


@retry(retry_on=_is_transient)
async def fetch_page(http: httpx.AsyncClient, title: str) -> dict[str, Any]:
    resp = await http.get(API_URL, params={**API_PARAMS, "titles": title})
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


async def fetch_guide(http: httpx.AsyncClient, city: str, guides_dir: Path) -> GuideStats:
    title, text = parse_page(await fetch_page(http, city))
    slug = slugify(city)
    # Fixed newline: the same article must give the same file on every OS.
    (guides_dir / f"{slug}.txt").write_text(text, encoding="utf-8", newline="\n")
    return GuideStats(
        slug=slug,
        title=title,
        chars=len(text),
        sections=len(ANY_HEADER.findall(text)),
        see_do_chars=see_do_size(text),
    )


def log_table(stats: list[GuideStats]) -> None:
    logger.info("%-12s %-20s %8s %9s %8s", "slug", "title", "chars", "sections", "see+do")
    for s in stats:
        logger.info(
            "%-12s %-20s %8d %9d %8d%s",
            s.slug, s.title, s.chars, s.sections, s.see_do_chars,
            "  <- THIN" if s.is_thin else "",
        )


async def main() -> int:
    guides_dir = get_settings().guides_dir
    guides_dir.mkdir(parents=True, exist_ok=True)
    stats: list[GuideStats] = []
    failed: list[str] = []
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=TIMEOUT_S, headers=headers) as http:
        for i, city in enumerate(CITIES):  # one article per request, sequential to be polite
            if i:
                await asyncio.sleep(PAUSE_S)
            try:
                stats.append(await fetch_guide(http, city, guides_dir))
            except (httpx.HTTPError, ValueError) as exc:
                logger.error("%s: fetch failed: %s", city, exc)
                failed.append(city)

    log_table(stats)
    thin = [s.slug for s in stats if s.is_thin]
    if thin:
        logger.warning("See + Do under %d characters: %s", MIN_SEE_DO_CHARS, ", ".join(thin))
    if failed:
        logger.error("not fetched: %s", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(main()))
