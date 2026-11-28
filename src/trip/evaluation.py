"""Running and scoring the extraction variants over the labelled dataset. Knows no SDK."""

import asyncio
import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from trip.dataset import DatasetItem
from trip.extract import Extractor
from trip.models import ParsedQuestion, TripError

FIELDS = tuple(ParsedQuestion.model_fields)


class Trace(BaseModel):
    """One call: what was asked of which variant, what came back and how it scored."""

    variant: str
    id: str
    run: int
    model: str
    error: Literal["format", "unavailable"] | None
    error_detail: str | None = None
    # The fields below are None when no reply was obtained.
    raw_text: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    retries: int | None = None
    prediction: ParsedQuestion | None = None  # also None on a format failure
    correct: dict[str, bool]


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    questions: int
    exact: int  # questions matched on every field in most of their answered runs
    fields: dict[str, int]  # the same count, field by field
    calls: int
    errors: int  # calls without a reply, left out of every other figure
    format_failures: int
    retried: int
    prompt_tokens: float | None  # mean per answered call
    completion_tokens: float | None
    p50_s: float | None  # answered calls that needed no retry
    p95_s: float | None
    flips: int  # questions whose answer changed between runs


@dataclass(frozen=True)
class Disagreement:
    item: DatasetItem
    matched: dict[str, bool]  # variant -> exact match
    answers: dict[str, list[ParsedQuestion | None]]  # variant -> distinct answers over the runs


def normalise(value: str) -> str:
    """Case and accents removed: "Kraków" and "krakow" are the same answer."""
    # Only combining marks are dropped: a letter like "Ł", which has none, must survive.
    decomposed = unicodedata.normalize("NFKD", value.strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def score_fields(expected: DatasetItem, predicted: ParsedQuestion | None) -> dict[str, bool]:
    """Field name -> whether it is right. Everything is wrong without a prediction."""
    if predicted is None:
        return dict.fromkeys(FIELDS, False)
    want, got = _comparable(expected), _comparable(predicted)
    return {name: got[name] == want[name] for name in FIELDS}


def _comparable(answer: DatasetItem | ParsedQuestion) -> dict[str, object]:
    values = {name: getattr(answer, name) for name in FIELDS}
    return {k: normalise(v) if isinstance(v, str) else v for k, v in values.items()}


def _answer_key(answer: ParsedQuestion | None) -> tuple[object, ...] | None:
    return None if answer is None else tuple(_comparable(answer).values())


async def run_eval(
    items: Sequence[DatasetItem],
    extractors: Mapping[str, Extractor],
    *,
    model: str,
    repeats: int,
    concurrency: int,
    out: Path,
) -> list[Trace]:
    """Call every extractor on every item `repeats` times, writing each trace to `out` at once.

    A call without a reply becomes a trace; any other error stops the run. `out` must not exist.
    """
    semaphore = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as sink:

        async def one(variant: str, item: DatasetItem, run: int) -> Trace:
            async with semaphore:
                trace = await _call(variant, extractors[variant], item, run, model)
            sink.write(trace.model_dump_json() + "\n")
            sink.flush()  # a crash must not lose the replies already paid for
            return trace

        async with asyncio.TaskGroup() as tg:
            # The variant varies fastest, so that all of them meet the same load and rate limits.
            tasks = [
                tg.create_task(one(variant, item, run))
                for run in range(1, repeats + 1)
                for item in items
                for variant in extractors
            ]
    return [task.result() for task in tasks]


async def _call(
    variant: str, extractor: Extractor, item: DatasetItem, run: int, model: str
) -> Trace:
    try:
        extraction = await extractor.extract(item.q, item.today)
    except TripError as exc:
        return Trace(
            variant=variant, id=item.id, run=run, model=model,
            error="unavailable", error_detail=repr(exc.__cause__ or exc),
            correct=score_fields(item, None),
        )
    reply = extraction.completion
    return Trace(
        variant=variant, id=item.id, run=run, model=model,
        error="format" if extraction.parsed is None else None,
        raw_text=reply.text,
        finish_reason=reply.finish_reason,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        latency_s=reply.latency_s,
        retries=reply.retries,
        prediction=extraction.parsed,
        correct=score_fields(item, extraction.parsed),
    )


def summarise(traces: Sequence[Trace], items: Sequence[DatasetItem]) -> list[VariantSummary]:
    """One summary per variant, in the order they were run."""
    summaries = []
    for variant, by_item in _group(traces).items():
        runs = [_answered(by_item.get(item.id, [])) for item in items]
        calls = [t for t in traces if t.variant == variant]
        answered = _answered(calls)
        p50, p95 = _percentiles(
            [t.latency_s for t in answered if t.retries == 0 and t.latency_s is not None]
        )
        summaries.append(
            VariantSummary(
                variant=variant,
                questions=len(items),
                exact=sum(_majority([_exact(t) for t in r]) for r in runs),
                fields={
                    name: sum(_majority([t.correct[name] for t in r]) for r in runs)
                    for name in FIELDS
                },
                calls=len(calls),
                errors=len(calls) - len(answered),
                format_failures=sum(t.error == "format" for t in answered),
                retried=sum(bool(t.retries) for t in answered),
                prompt_tokens=_mean([t.prompt_tokens for t in answered]),
                completion_tokens=_mean([t.completion_tokens for t in answered]),
                p50_s=p50,
                p95_s=p95,
                flips=sum(len({_answer_key(t.prediction) for t in r}) > 1 for r in runs),
            )
        )
    return summaries


def exact_matches(traces: Sequence[Trace]) -> dict[str, dict[str, bool]]:
    """variant -> question ID -> whether most of its answered runs matched on every field."""
    return {
        variant: {
            item_id: _majority([_exact(t) for t in _answered(runs)])
            for item_id, runs in by_item.items()
        }
        for variant, by_item in _group(traces).items()
    }


def exact_by_tag(
    traces: Sequence[Trace], items: Sequence[DatasetItem]
) -> dict[str, tuple[int, dict[str, int]]]:
    """tag -> (number of questions with it, variant -> number of exact matches among them)."""
    matches = exact_matches(traces)
    table = {}
    for tag in sorted({tag for item in items for tag in item.tags}):
        tagged = [item.id for item in items if tag in item.tags]
        scores = {v: sum(found.get(i, False) for i in tagged) for v, found in matches.items()}
        table[tag] = (len(tagged), scores)
    return table


def disagreements(traces: Sequence[Trace], items: Sequence[DatasetItem]) -> list[Disagreement]:
    """The questions that at least one variant gets right and another wrong."""
    matches = exact_matches(traces)
    grouped = _group(traces)
    found = []
    for item in items:
        matched = {variant: by_id.get(item.id, False) for variant, by_id in matches.items()}
        if len(set(matched.values())) < 2:
            continue
        answers = {}
        for variant, by_item in grouped.items():
            runs = sorted(_answered(by_item.get(item.id, [])), key=lambda t: t.run)
            distinct = {_answer_key(t.prediction): t.prediction for t in runs}
            answers[variant] = list(distinct.values())
        found.append(Disagreement(item, matched, answers))
    return found


def _group(traces: Sequence[Trace]) -> dict[str, dict[str, list[Trace]]]:
    grouped: dict[str, dict[str, list[Trace]]] = {}
    for trace in traces:
        grouped.setdefault(trace.variant, {}).setdefault(trace.id, []).append(trace)
    return grouped


def _answered(traces: Sequence[Trace]) -> list[Trace]:
    return [t for t in traces if t.error != "unavailable"]


def _exact(trace: Trace) -> bool:
    return all(trace.correct.values())


def _majority(outcomes: Sequence[bool]) -> bool:
    """True in more than half of the runs; False without any run."""
    return 2 * sum(outcomes) > len(outcomes)


def _mean(values: Sequence[int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return statistics.fmean(known) if known else None


def _percentiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:  # quantiles() needs two points
        return (values[0], values[0]) if values else (None, None)
    # "inclusive" never extrapolates past the slowest call observed.
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return cuts[49], cuts[94]
