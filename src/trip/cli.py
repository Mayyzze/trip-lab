import argparse
import logging
import sys

from mistralai.client import Mistral

from trip.chat import TripService
from trip.config import get_settings
from trip.llm import MistralLLM
from trip.models import TripError

logger = logging.getLogger(__name__)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trip")
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs")
    args = parser.parse_args(argv)

    logging.basicConfig(             
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx loggue chaque requête en INFO

    settings = get_settings()
    mistral = MistralLLM(client=Mistral(api_key=settings.mistral_api_key.get_secret_value()), settings=settings)
    trip_svc = TripService(llm=mistral, horizon_days=get_settings().forecast_horizon_days)
    try:
        ans = trip_svc.answer(args.question)
    except TripError as exc:
        logger.debug("request failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(ans)
    return 0