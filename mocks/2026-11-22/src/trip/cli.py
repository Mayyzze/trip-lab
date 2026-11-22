import argparse
import logging
import sys

from trip.chat import answer
from trip.models import TripError, Units

logger = logging.getLogger(__name__)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trip")
    parser.add_argument("question")
    parser.add_argument("--units", type=Units, choices=list(Units), default=Units.METRIC)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs")
    args = parser.parse_args(argv)

    logging.basicConfig(             
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx loggue chaque requête en INFO

    try:
        ans = answer(args.question, args.unit)
    except TripError as exc:
        logger.debug("request failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(ans)
    return 0