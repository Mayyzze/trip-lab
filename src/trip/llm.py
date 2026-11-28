import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from itertools import batched
from typing import Any

import httpx2  # the HTTP library of the Mistral SDK: its network errors are httpx2 errors
from mistralai.client import Mistral
from mistralai.client.errors import MistralError, NoResponseError
from mistralai.client.models import ChatCompletionResponse
from mistralai.extra.utils.response_format import response_format_from_pydantic_model
from pydantic import BaseModel

from trip.config import Settings
from trip.models import (
    Completion,
    EmbeddingUnavailable,
    Evidence,
    LLMUnavailable,
    Message,
    TripError,
)
from trip.prompts import GUIDE_PROMPT, WEATHER_PROMPT, format_sources
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

    def __init__(
        self,
        *,
        client: Mistral,
        settings: Settings,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.client = client
        self.settings = settings
        self._sleep = sleep  # injected so that tests do not wait between retries

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        response_schema: type[BaseModel] | None = None,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.settings.mistral_model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        if response_schema:
            # What chat.parse_async sends. Parsing is left to the caller: parse_async raises on a
            # truncated reply and loses its token counts and finish reason.
            request["response_format"] = response_format_from_pydantic_model(response_schema)

        attempts = 0
        latency_s = 0.0

        @retry(retry_on=_is_transient, sleep=self._sleep)
        async def call() -> ChatCompletionResponse:
            nonlocal attempts, latency_s
            attempts += 1
            started = time.perf_counter()  # inside the retry: backoff is not model latency
            response = await self.client.chat.complete_async(**request)
            latency_s = time.perf_counter() - started
            return response

        try:
            res = await call()
        except (MistralError, NoResponseError, httpx2.HTTPError) as exc:
            logger.warning("chat call failed after %d attempt(s): %s", attempts, exc)
            raise LLMUnavailable("language model unavailable") from exc
        if not res.choices:
            raise LLMUnavailable("the model returned no choice")

        choice = res.choices[0]
        content = choice.message.content if choice.message else None
        return Completion(
            text=content if isinstance(content, str) else "",
            prompt_tokens=res.usage.prompt_tokens or 0,
            completion_tokens=res.usage.completion_tokens or 0,
            finish_reason=str(choice.finish_reason),
            latency_s=latency_s,
            retries=attempts - 1,
        )

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
