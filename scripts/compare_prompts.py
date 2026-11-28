"""Compare the extraction prompt variants on evals/dataset.jsonl. Every call is paid.

Usage: uv run python scripts/compare_prompts.py [--variants NAME ...] [--repeats 3]
                                                [--limit N] [--concurrency 4]
Calls = questions x variants x repeats. Traces go to evals/results/.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import datetime

from mistralai.client import Mistral

from trip.config import PROJECT_ROOT, get_settings
from trip.dataset import DatasetItem, load_dataset
from trip.evaluation import (
    FIELDS,
    Trace,
    disagreements,
    exact_by_tag,
    run_eval,
    summarise,
)
from trip.extract import VARIANTS
from trip.llm import MistralLLM
from trip.models import ParsedQuestion

DATASET = PROJECT_ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the extraction prompt variants.")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--limit", type=positive_int, help="first N questions only")
    parser.add_argument("--concurrency", type=positive_int, default=4)
    return parser.parse_args(argv)


def print_summary(traces: Sequence[Trace], items: Sequence[DatasetItem]) -> None:
    n = len(items)
    header = f"{'variant':<11} {'exact':>11}"
    header += "".join(f" {name:>7}" for name in FIELDS)
    header += f" {'format':>7} {'errors':>6} {'retried':>7} {'in tok':>6} {'out tok':>7}"
    header += f" {'p50':>6} {'p95':>6} {'flips':>6}"
    print(f"\nPer question: right in more than half of its runs (n = {n})")
    print(header)
    for s in summarise(traces, items):
        row = f"{s.variant:<11} {f'{s.exact}/{n}':>6} {s.exact / n:>4.0%}"
        row += "".join(f" {f'{s.fields[name]}/{n}':>7}" for name in FIELDS)
        row += f" {f'{s.format_failures}/{s.calls - s.errors}':>7}"
        row += f" {s.errors:>6} {s.retried:>7}"
        row += f" {_number(s.prompt_tokens):>6} {_number(s.completion_tokens):>7}"
        row += f" {_seconds(s.p50_s):>6} {_seconds(s.p95_s):>6} {f'{s.flips}/{n}':>6}"
        print(row)
    print(f"1 question = {100 / n:.1f} points: a gap of 1 or 2 questions is noise.")


def print_tags(traces: Sequence[Trace], items: Sequence[DatasetItem]) -> None:
    variants = list(dict.fromkeys(t.variant for t in traces))
    print("\nExact match by tag")
    print(f"{'tag':<15} {'n':>3}" + "".join(f" {v:>11}" for v in variants))
    for tag, (total, scores) in exact_by_tag(traces, items).items():
        cells = "".join(f" {f'{scores[v]}/{total}':>11}" for v in variants)
        print(f"{tag:<15} {total:>3}{cells}")


def print_disagreements(traces: Sequence[Trace], items: Sequence[DatasetItem]) -> None:
    found = disagreements(traces, items)
    print(f"\nDisagreements: {len(found)} question(s)")
    print(f"answers read: {' | '.join(FIELDS)}")
    for d in found:
        print(f"\n{d.item.id} (today {d.item.today:%a %Y-%m-%d}) {d.item.q}")
        print(f"  {'expected':<17} {_answer(d.item)}")
        for variant, matched in d.matched.items():
            answers = " ; ".join(_answer(a) for a in d.answers[variant]) or "no reply"
            print(f"  {variant:<11} {'ok' if matched else 'WRONG':<5} {answers}")


def _answer(answer: DatasetItem | ParsedQuestion | None) -> str:
    if answer is None:
        return "format failure"
    values = (getattr(answer, name) for name in FIELDS)
    return " | ".join("-" if v is None else str(v) for v in values)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # A redirected Windows output is cp1252: a character it lacks must not lose the report.
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    items = load_dataset(DATASET)[: args.limit]
    client = Mistral(api_key=settings.mistral_api_key.get_secret_value())
    llm = MistralLLM(client=client, settings=settings)
    extractors = {name: VARIANTS[name](llm) for name in args.variants}
    out = RESULTS_DIR / f"s5-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print(
        f"{len(items) * len(extractors) * args.repeats} calls to {settings.mistral_model}: "
        f"{len(items)} questions x {len(extractors)} variants x {args.repeats} runs",
        flush=True,
    )

    traces = asyncio.run(
        run_eval(
            items,
            extractors,
            model=settings.mistral_model,
            repeats=args.repeats,
            concurrency=args.concurrency,
            out=out,
        )
    )
    print_summary(traces, items)
    print_tags(traces, items)
    print_disagreements(traces, items)
    print(f"\ntraces: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
