"""The extraction variants: what each one sends and how it reads the reply."""

from datetime import date
from typing import get_args

import pytest
from pydantic import ValidationError

from trip.config import PROJECT_ROOT, Settings
from trip.dataset import load_dataset
from trip.extract import VARIANTS, parse_json_answer
from trip.models import ExtractionFormatError, ParsedQuestion
from trip.prompts import COT_INSTRUCTION, EXTRACTION_RULES, FEW_SHOT_EXAMPLES

from .fakes import FakeLLM, completion

TODAY = date(2026, 12, 2)  # a Wednesday
QUESTION = "Weather in Lisbon tomorrow?"
EXPECTED = ParsedQuestion(
    city="Lisbon", start=date(2026, 12, 3), end=date(2026, 12, 3), topic="weather"
)
JSON = EXPECTED.model_dump_json()
DRAFT = '{"city": "Porto", "start": null, "end": null, "topic": "both"}'


@pytest.mark.parametrize(
    "text",
    [
        JSON,
        f"```json\n{JSON}\n```",
        f"Here it is: {JSON}\nHope this helps.",
        f"Draft: {DRAFT}\nFinal: {JSON}",
        f"A first guess {DRAFT}, but tomorrow is Thursday.\nANSWER: {JSON}",
        f"ANSWER:\n```json\n{JSON}\n```",
        f"The set {{city, start}} is needed. {JSON}",
    ],
    ids=["plain", "fenced", "trailing_text", "last_object", "marker", "marker_fenced", "braces"],
)
def test_the_parser_finds_the_answer(text: str):
    assert parse_json_answer(text) == EXPECTED


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I do not know.",
        '{"city": "Lisbon", "start": null,}',
        '{"city": 3, "start": null, "end": null, "topic": "weather"}',
        '{"city": "Lisbon", "start": "tomorrow", "end": null, "topic": "weather"}',
        '{"city": "Lisbon", "start": null, "end": null, "topic": "food"}',
        '{"city": "Lisbon", "topic": "weather"}',
        f"{JSON}\nANSWER: none",
    ],
    ids=[
        "empty", "no_object", "invalid_json", "wrong_type", "bad_date", "bad_topic",
        "missing_keys", "nothing_after_marker",
    ],
)
def test_the_parser_refuses_an_unusable_answer(text: str):
    with pytest.raises(ExtractionFormatError):
        parse_json_answer(text)


@pytest.mark.parametrize("name", VARIANTS)
async def test_every_variant_sends_the_rules_and_the_weekday(name: str):
    llm = FakeLLM(reply=completion(JSON))

    result = await VARIANTS[name](llm).extract(QUESTION, TODAY)

    [request] = llm.completed
    system, last = request.messages[0], request.messages[-1]
    assert system["role"] == "system" and EXTRACTION_RULES in system["content"]
    assert last == {"role": "user", "content": f"Today is Wednesday 2026-12-02.\nQuestion: {QUESTION}"}
    assert result.parsed == EXPECTED


async def test_each_variant_changes_only_its_technique():
    requests = {}
    for name, build in VARIANTS.items():
        llm = FakeLLM(reply=completion(JSON))
        await build(llm).extract(QUESTION, TODAY)
        [requests[name]] = llm.completed

    zero = requests["zero_shot"]
    few = requests["few_shot"]
    cot = requests["cot"]
    structured = requests["structured"]

    assert few.messages[0] == zero.messages[0] and few.messages[-1] == zero.messages[-1]
    assert len(few.messages) == len(zero.messages) + 2 * len(FEW_SHOT_EXAMPLES)
    assert cot.messages[0]["content"] == f"{EXTRACTION_RULES}\n\n{COT_INSTRUCTION}"
    assert cot.messages[1:] == zero.messages[1:]
    assert structured.messages == zero.messages
    assert [r.response_schema for r in (zero, few, cot, structured)] == [
        None, None, None, ParsedQuestion
    ]
    assert cot.max_tokens > zero.max_tokens == few.max_tokens == structured.max_tokens


@pytest.mark.parametrize("finish_reason", ["length", "model_length"])
@pytest.mark.parametrize("name", VARIANTS)
async def test_a_truncated_reply_is_a_failure_even_if_it_parses(name: str, finish_reason: str):
    llm = FakeLLM(reply=completion(JSON, finish_reason))

    result = await VARIANTS[name](llm).extract(QUESTION, TODAY)

    assert result.parsed is None
    assert result.completion.text == JSON


@pytest.mark.parametrize("name", VARIANTS)
async def test_an_unreadable_reply_keeps_its_cost(name: str):
    llm = FakeLLM(reply=completion("Sorry."))

    result = await VARIANTS[name](llm).extract(QUESTION, TODAY)

    assert result.parsed is None
    assert (result.completion.prompt_tokens, result.completion.completion_tokens) == (100, 20)


async def test_structured_does_not_repair_a_reply_that_ignores_the_schema():
    llm = FakeLLM(reply=completion(f"```json\n{JSON}\n```"))

    assert (await VARIANTS["structured"](llm).extract(QUESTION, TODAY)).parsed is None


def test_the_rules_contain_the_labelling_conventions_verbatim():
    document = (PROJECT_ROOT / "evals" / "LABELLING.md").read_text(encoding="utf-8")
    _, _, rest = document.partition("<!-- conventions:start -->")
    block, end_marker, _ = rest.partition("<!-- conventions:end -->")

    assert end_marker and block.strip()
    assert block.strip() in EXTRACTION_RULES


def test_the_rules_name_every_field_of_the_schema():
    for field in ParsedQuestion.model_fields:
        assert f'"{field}":' in EXTRACTION_RULES


def test_the_configurable_variants_are_the_implemented_ones():
    annotation = Settings.model_fields["extraction_variant"].annotation

    assert set(get_args(annotation)) == set(VARIANTS)


def test_the_chosen_variant_is_the_default_and_a_typo_is_refused():
    assert Settings.model_fields["extraction_variant"].default == "few_shot"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mistral_api_key="k", extraction_variant="fewshot")  # type: ignore[call-arg, arg-type]


def test_no_few_shot_example_leaks_from_the_dataset():
    items = load_dataset(PROJECT_ROOT / "evals" / "dataset.jsonl")
    questions = {item.q.casefold() for item in items}
    cities = {item.city for item in items if item.city}
    for _, question, answer in FEW_SHOT_EXAMPLES:
        assert question.casefold() not in questions
        # Not only absent as a city label: a two-city question has none.
        assert answer.city not in cities
        assert not any(answer.city and answer.city in item.q for item in items)
