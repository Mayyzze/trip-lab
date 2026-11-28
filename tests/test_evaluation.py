"""Scoring, aggregation and the eval run, offline."""

import asyncio
from datetime import date
from pathlib import Path

import pytest

from trip.dataset import DatasetItem
from trip.evaluation import (
    FIELDS,
    Trace,
    disagreements,
    exact_by_tag,
    normalise,
    run_eval,
    score_fields,
    summarise,
)
from trip.extract import VARIANTS, Extraction
from trip.models import LLMUnavailable, ParsedQuestion

from .fakes import FakeLLM, completion

TODAY = date(2026, 12, 2)


def item(item_id: str, city: str = "Krakow", *, tags: tuple[str, ...] = ()) -> DatasetItem:
    return DatasetItem(
        id=item_id, q=f"Weather in {city}?", today=TODAY, city=city, start=None, end=None,
        topic="weather", lang="en", tags=frozenset(tags),
    )


def answer(city: str | None = "Krakow", **changes: object) -> ParsedQuestion:
    fields = {"city": city, "start": None, "end": None, "topic": "weather", **changes}
    return ParsedQuestion.model_validate(fields)


def trace(
    expected: DatasetItem,
    prediction: ParsedQuestion | None,
    *,
    variant: str = "zero_shot",
    run: int = 1,
    latency_s: float = 1.0,
    retries: int = 0,
    unavailable: bool = False,
) -> Trace:
    if unavailable:
        return Trace(
            variant=variant, id=expected.id, run=run, model="m", error="unavailable",
            correct=score_fields(expected, None),
        )
    return Trace(
        variant=variant, id=expected.id, run=run, model="m",
        error="format" if prediction is None else None,
        prompt_tokens=100, completion_tokens=20, latency_s=latency_s, retries=retries,
        prediction=prediction, correct=score_fields(expected, prediction),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("Kraków", "krakow"), (" KRAKOW ", "krakow"), ("Île-de-France", "ile-de-france"),
     ("Łódź", "łodz")],
)
def test_normalise_ignores_case_and_accents_only(value: str, expected: str):
    assert normalise(value) == expected


def test_all_six_fields_are_scored():
    assert FIELDS == ("city", "start", "end", "topic", "country", "region")
    assert all(score_fields(item("a"), answer("kraków")).values())


@pytest.mark.parametrize(
    ("prediction", "wrong"),
    [
        (answer("Cracow"), {"city"}),
        (answer(None), {"city"}),
        (answer(start="2026-12-03"), {"start"}),
        (answer(topic="both"), {"topic"}),
        (answer(country="PL"), {"country"}),  # invented: the label is null
        (None, set(FIELDS)),
    ],
)
def test_a_wrong_field_is_reported_alone(prediction: ParsedQuestion | None, wrong: set[str]):
    correct = score_fields(item("a"), prediction)

    assert {name for name, ok in correct.items() if not ok} == wrong


def test_a_question_counts_if_most_of_its_runs_match():
    mostly, rarely = item("a"), item("b")
    right, wrong = answer(), answer("Porto")
    traces = [
        trace(mostly, right, run=1), trace(mostly, right, run=2), trace(mostly, wrong, run=3),
        trace(rarely, right, run=1), trace(rarely, wrong, run=2), trace(rarely, wrong, run=3),
    ]

    [summary] = summarise(traces, [mostly, rarely])

    assert (summary.exact, summary.questions, summary.flips) == (1, 2, 2)
    assert summary.fields["city"] == 1 and summary.fields["topic"] == 2


def test_a_spelling_difference_is_not_a_flip():
    a = item("a")
    traces = [trace(a, answer("Krakow"), run=1), trace(a, answer("Kraków"), run=2)]

    assert summarise(traces, [a])[0].flips == 0


def test_a_format_failure_is_wrong_and_counted_apart():
    a = item("a")
    traces = [trace(a, None, run=1), trace(a, None, run=2), trace(a, answer(), run=3)]

    [summary] = summarise(traces, [a])

    assert (summary.exact, summary.format_failures, summary.flips) == (0, 2, 1)


def test_calls_without_a_reply_are_left_out_of_the_figures():
    a, never = item("a"), item("b")
    traces = [
        trace(a, answer(), run=1),
        trace(a, None, run=2, unavailable=True),
        trace(never, None, run=1, unavailable=True),
    ]

    [summary] = summarise(traces, [a, never])

    assert (summary.exact, summary.questions) == (1, 2)
    assert (summary.calls, summary.errors, summary.format_failures, summary.flips) == (3, 2, 0, 0)
    assert summary.prompt_tokens == 100


def test_retried_calls_are_counted_and_left_out_of_the_latency():
    a = item("a")
    traces = [
        trace(a, answer(), run=1, latency_s=1.0),
        trace(a, answer(), run=2, latency_s=3.0),
        trace(a, answer(), run=3, latency_s=60.0, retries=1),
    ]

    [summary] = summarise(traces, [a])

    assert summary.retried == 1
    assert summary.p50_s == 2.0
    assert summary.p95_s is not None and summary.p95_s <= 3.0


def test_exact_matches_are_broken_down_by_tag():
    a = item("a", tags=("typo", "no_date"))
    b = item("b", tags=("no_date",))
    traces = [
        trace(a, answer()), trace(b, answer("Porto")),
        trace(a, answer("Porto"), variant="cot"), trace(b, answer(), variant="cot"),
    ]

    assert exact_by_tag(traces, [a, b]) == {
        "no_date": (2, {"zero_shot": 1, "cot": 1}),
        "typo": (1, {"zero_shot": 1, "cot": 0}),
    }


def test_disagreements_keep_only_the_questions_that_split_the_variants():
    split, agreed = item("a"), item("b")
    traces = [
        trace(split, answer()), trace(agreed, answer()),
        trace(split, None, variant="cot"), trace(agreed, answer(), variant="cot"),
    ]

    [found] = disagreements(traces, [split, agreed])

    assert found.item == split
    assert found.matched == {"zero_shot": True, "cot": False}
    assert found.answers == {"zero_shot": [answer()], "cot": [None]}


async def test_a_run_writes_one_trace_line_per_call(tmp_path: Path):
    items = [item("a"), item("b", "Porto")]
    llm = FakeLLM(reply=completion(answer().model_dump_json()))
    extractors = {name: VARIANTS[name](llm) for name in ("zero_shot", "structured")}
    out = tmp_path / "results" / "run.jsonl"

    traces = await run_eval(items, extractors, model="m", repeats=2, concurrency=1, out=out)

    written = [Trace.model_validate_json(line) for line in out.read_text("utf-8").splitlines()]
    assert written == traces
    assert [(t.run, t.id, t.variant) for t in written] == [
        (run, item_id, variant)
        for run in (1, 2)
        for item_id in ("a", "b")
        for variant in ("zero_shot", "structured")
    ]
    first, porto = written[0], written[2]
    assert first.error is None and all(first.correct.values())
    assert (first.model, first.finish_reason, first.retries) == ("m", "stop", 0)
    assert (first.prompt_tokens, first.completion_tokens, first.latency_s) == (100, 20, 0.5)
    assert first.raw_text == answer().model_dump_json()
    assert not porto.correct["city"] and porto.correct["topic"]


class StubExtractor:
    """Raises `error`, or reports an unusable reply; records how many calls overlap."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.in_flight = 0
        self.max_in_flight = 0

    async def extract(self, question: str, today: date) -> Extraction:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            if self._error:
                raise self._error
            return Extraction(None, completion("?"))
        finally:
            self.in_flight -= 1


async def test_a_run_never_exceeds_its_concurrency(tmp_path: Path):
    stub = StubExtractor()
    items = [item(str(i)) for i in range(10)]

    traces = await run_eval(
        items, {"stub": stub}, model="m", repeats=1, concurrency=3, out=tmp_path / "run.jsonl"
    )

    assert stub.max_in_flight == 3
    assert {t.error for t in traces} == {"format"}


async def test_an_unavailable_model_is_recorded_and_the_run_goes_on(tmp_path: Path):
    extractors = {"down": StubExtractor(LLMUnavailable("down")), "up": StubExtractor()}

    traces = await run_eval(
        [item("a")], extractors, model="m", repeats=1, concurrency=2, out=tmp_path / "run.jsonl"
    )

    assert [t.error for t in traces] == ["unavailable", "format"]
    assert "down" in (traces[0].error_detail or "")


async def test_an_unexpected_error_stops_the_run(tmp_path: Path):
    with pytest.raises(ExceptionGroup):
        await run_eval(
            [item("a")], {"bug": StubExtractor(RuntimeError("bug"))},
            model="m", repeats=1, concurrency=1, out=tmp_path / "run.jsonl",
        )
