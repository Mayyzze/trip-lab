"""The prompt variants that turn a question into a ParsedQuestion. Knows nothing about any SDK."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from pydantic import BaseModel, ValidationError

from trip.models import Completion, ExtractionFormatError, Message, ParsedQuestion
from trip.ports import ChatModel
from trip.prompts import (
    COT_INSTRUCTION,
    EXTRACTION_RULES,
    FEW_SHOT_EXAMPLES,
    format_question,
)

ANSWER_MARKER = "ANSWER:"
# "model_length" is a cut at the context limit instead of max_tokens: the reply is just as partial.
TRUNCATED = frozenset({"length", "model_length"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Extraction:
    parsed: ParsedQuestion | None  # None is a format failure; the reply is kept for the traces
    completion: Completion


class Extractor(Protocol):
    async def extract(self, question: str, today: date) -> Extraction:
        """Extract the city, dates and topic of a question asked on `today`.

        A truncated or unreadable reply gives `parsed=None`.
        Raises LLMUnavailable if no reply can be obtained.
        """
        ...


class ZeroShotExtractor:
    """The shared rules and the question. The other variants each change one thing."""

    max_tokens = 300  # the JSON takes under 100 tokens; the rest is room for a preamble
    response_schema: type[BaseModel] | None = None

    def __init__(self, llm: ChatModel) -> None:
        self._llm = llm

    def messages(self, question: str, today: date) -> list[Message]:
        return [
            {"role": "system", "content": EXTRACTION_RULES},
            {"role": "user", "content": format_question(question, today)},
        ]

    def parse(self, text: str) -> ParsedQuestion:
        return parse_json_answer(text)

    async def extract(self, question: str, today: date) -> Extraction:
        completion = await self._llm.complete(
            self.messages(question, today),
            max_tokens=self.max_tokens,
            response_schema=self.response_schema,
        )
        return Extraction(self._read(completion), completion)

    def _read(self, completion: Completion) -> ParsedQuestion | None:
        # Checked before parsing: a cut reply can still contain an earlier, complete object.
        if completion.finish_reason in TRUNCATED:
            logger.warning("truncated extraction (%s)", completion.finish_reason)
            return None
        try:
            return self.parse(completion.text)
        except ExtractionFormatError as exc:
            logger.warning("unusable extraction: %s", exc)
            return None


class FewShotExtractor(ZeroShotExtractor):
    def messages(self, question: str, today: date) -> list[Message]:
        system, user = super().messages(question, today)
        examples: list[Message] = []
        for example_today, example_question, answer in FEW_SHOT_EXAMPLES:
            examples.append(
                {"role": "user", "content": format_question(example_question, example_today)}
            )
            examples.append({"role": "assistant", "content": answer.model_dump_json()})
        return [system, *examples, user]


class CotExtractor(ZeroShotExtractor):
    max_tokens = 800  # the reasoning comes before the JSON

    def messages(self, question: str, today: date) -> list[Message]:
        system, user = super().messages(question, today)
        content = f"{system['content']}\n\n{COT_INSTRUCTION}"
        return [{"role": "system", "content": content}, user]


class StructuredExtractor(ZeroShotExtractor):
    """The API constrains the reply to the ParsedQuestion schema."""

    response_schema = ParsedQuestion

    def parse(self, text: str) -> ParsedQuestion:
        # Strict on purpose: the tolerant parser would hide a reply that ignored the schema.
        return _validate(text)


VARIANTS: dict[str, Callable[[ChatModel], Extractor]] = {
    "zero_shot": ZeroShotExtractor,
    "few_shot": FewShotExtractor,
    "cot": CotExtractor,
    "structured": StructuredExtractor,
}


def parse_json_answer(text: str) -> ParsedQuestion:
    """The last JSON object after "ANSWER:", or in the whole text without that marker.

    Raises ExtractionFormatError if there is none or if its fields are invalid.
    """
    candidate = _last_json_object(text.rsplit(ANSWER_MARKER, 1)[-1])
    if candidate is None:
        raise ExtractionFormatError("no JSON object in the reply")
    return _validate(candidate)


def _validate(source: str) -> ParsedQuestion:
    try:
        return ParsedQuestion.model_validate_json(source)
    except ValidationError as exc:
        raise ExtractionFormatError(f"invalid extraction: {exc}") from exc


def _last_json_object(text: str) -> str | None:
    """The source of the last outermost JSON object in `text`."""
    decoder = json.JSONDecoder()
    last = None
    start = text.find("{")
    while start != -1:
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            end = start + 1  # a brace in prose
        else:
            last = text[start:end]
        start = text.find("{", end)  # resuming after an object skips the ones nested in it
    return last
