import json
import logging
from datetime import date
from itertools import batched

import httpx2  # the HTTP library of the Mistral SDK: its network errors are httpx2 errors
from mistralai.client import Mistral
from mistralai.client.errors import MistralError, NoResponseError
from pydantic import ValidationError

from trip.config import Settings
from trip.models import EmbeddingUnavailable, Evidence, ParsedQuestion, TripError
from trip.prompts import EXTRACT_PROMPT, GUIDE_PROMPT, WEATHER_PROMPT, format_sources
from trip.retry import retry

logger = logging.getLogger(__name__)

def _is_transient(exc: Exception) -> bool:
    """Errors worth a second try: timeouts, network failures, 429 and 5xx."""
    if isinstance(exc, MistralError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(
        exc, httpx2.TimeoutException | httpx2.NetworkError | httpx2.RemoteProtocolError
    )


class MistralEmbedder:

    def __init__(self, *, client: Mistral, model: str, batch_size: int):
        self._client = client
        self._model = model
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(texts, self._batch_size):  # sequential, to respect rate limits
            try:
                vectors += await self._embed_batch(list(batch))
            except (MistralError, NoResponseError, httpx2.HTTPError) as exc:
                logger.warning("embedding call failed: %s", exc)
                raise EmbeddingUnavailable("embedding service unavailable") from exc
        return vectors

    @retry(retry_on=_is_transient)
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        res = await self._client.embeddings.create_async(model=self._model, inputs=batch)
        logger.info("embedded %d texts, %s tokens", len(batch), res.usage.total_tokens)
        ordered = sorted(res.data, key=lambda item: item.index or 0)
        vectors = [item.embedding for item in ordered if item.embedding]
        if len(vectors) != len(batch):
            raise EmbeddingUnavailable(f"expected {len(batch)} embeddings, got {len(vectors)}")
        return vectors


class MistralLLM:

    def __init__(self, *,  client: Mistral, settings: Settings):
        self.client = client
        self.settings = settings

    async def parse_question(self, question: str, today: date) -> ParsedQuestion:
        try:
            res = await self.client.chat.parse_async(
                model=self.settings.mistral_model,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT.format(today=today)},
                    {"role": "user", "content": question},
                ],
                response_format=ParsedQuestion,
                temperature=0,
                top_p=1,
                max_tokens=200,
            )
        except ValidationError as exc:  # e.g. a date that is not a date
            raise TripError("could not understand the question") from exc

        choice = res.choices[0] if res.choices else None
        message = choice.message if choice else None
        parsed = message.parsed if message else None
        if parsed is None:
            raise TripError("could not understand the question")
        return parsed

    async def write_answer(self, question: str, evidence: Evidence) -> str:
        forecast = "not available"
        if evidence.forecast:
            loc, days = evidence.forecast
            forecast = json.dumps({
                "location": f"{loc.name}, {loc.country}",
                "forecast": [d.model_dump(mode="json") for d in days],
            })
        prompt = f"Question: {question}\n\nForecast JSON:\n{forecast}"
        if evidence.sources:
            prompt += f"\n\nSources:\n{format_sources(evidence.sources)}"

        res = await self.client.chat.complete_async(
            model=self.settings.mistral_model,
            messages=[
                {"role": "system", "content": GUIDE_PROMPT if evidence.sources else WEATHER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=700 if evidence.sources else 400,
        )
        content = res.choices[0].message.content if res.choices else None # type: ignore
        if not isinstance(content, str) or not content.strip():  # content can also be a list of chunks
            raise TripError("the model returned an empty answer")
        return content
