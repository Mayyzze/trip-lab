import argparse
import asyncio
import logging
import sys

import httpx
from mistralai.client import Mistral

from trip.cache import CachedWeather
from trip.config import get_settings
from trip.llm import MistralEmbedder, MistralLLM
from trip.models import AmbiguousCity, TripError
from trip.ports import WeatherProvider
from trip.rag import Retriever, load_or_build
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
    client = Mistral(api_key=settings.mistral_api_key.get_secret_value())
    mistral = MistralLLM(client=client, settings=settings)
    embedder = MistralEmbedder(
        client=client, model=settings.embed_model, batch_size=settings.embed_batch_size
    )

    async def core(question: str) -> None:
        index = await load_or_build(settings.guides_dir, settings.index_dir, embedder, settings)
        async with httpx.AsyncClient(timeout=5.0) as http:
            weather: WeatherProvider = OpenMeteoProvider(http)
            if settings.weather_cache_enabled:
                weather = CachedWeather(weather, forecast_ttl_s=settings.forecast_cache_ttl_s)
            service = TripService(
                weather,
                llm=mistral,
                guides=Retriever(index, embedder),
                horizon_days=settings.forecast_horizon_days,
                top_k=settings.rag_top_k,
            )
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
