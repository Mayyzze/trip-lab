import argparse
import asyncio
import logging
import sys

import httpx
from mistralai.client import Mistral

from trip.config import get_settings
from trip.llm import MistralLLM
from trip.models import AmbiguousCity, TripError
from trip.services import TripService
from trip.weather import OpenMeteoProvider

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
    
    
    async def core(question: str) -> None:
        async with httpx.AsyncClient(timeout=5.0) as http:
            service = TripService(OpenMeteoProvider(http), llm=mistral, horizon_days=get_settings().forecast_horizon_days)
            print(await service.answer(question))
        
    
    try:
        asyncio.run(core(args.question))
    except AmbiguousCity as exc:
        options = "\n".join(f"  - {c.label()}" for c in exc.candidates)
        print(f"Which {exc.city}?\n{options}\nAdd the region or country to your question.", file=sys.stderr)
        return 1            
    except TripError as exc:
        logger.debug("request failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
