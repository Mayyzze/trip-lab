"""Retrieval evaluation over evals/retrieval.jsonl: hit rate@k and MRR@k.

Usage: uv run python evals/run_retrieval.py
Labels are checked first, so a wrong label costs no embedding call.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from mistralai.client import Mistral
from pydantic import BaseModel, ConfigDict, Field

from trip.config import Settings, get_settings
from trip.llm import MistralEmbedder
from trip.models import TripError
from trip.rag import Chunk, Retriever, chunk, load_or_build, read_guides

LABELS_FILE = Path(__file__).with_name("retrieval.jsonl")

logger = logging.getLogger("run_retrieval")


class Label(BaseModel):
    """A question and the passage that answers it, independent of chunk IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    q: str = Field(min_length=1)
    city: str = Field(min_length=1)  # guide slug
    must_contain: str = Field(min_length=1)

    def matches(self, chunk: Chunk) -> bool:
        return chunk.city == self.city and self.must_contain.casefold() in chunk.text.casefold()


@dataclass(frozen=True)
class Result:
    label: Label
    rank: int | None  # 1-based rank of the first matching hit, None if missed
    best_score: float | None  # score of the first hit, None if there is none


def read_labels(path: Path) -> list[Label]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Label.model_validate_json(line) for line in lines if line.strip()]


def find_label_errors(labels: list[Label], chunks: list[Chunk]) -> list[Label]:
    """Labels that no chunk can satisfy: they would count as misses whatever the retriever does."""
    return [label for label in labels if not any(label.matches(c) for c in chunks)]


async def evaluate(labels: list[Label], retriever: Retriever, k: int) -> list[Result]:
    results: list[Result] = []
    for label in labels:
        hits = await retriever.search(label.q, k=k, city=label.city)
        rank = next((i for i, hit in enumerate(hits, 1) if label.matches(hit.chunk)), None)
        results.append(Result(label, rank, hits[0].score if hits else None))
    return results


def log_report(results: list[Result], k: int) -> None:
    logger.info("%-4s %4s %6s  %-10s %s", "", "rank", "best", "city", "question")
    for r in results:
        logger.info(
            "%-4s %4s %6s  %-10s %s",
            "HIT" if r.rank else "MISS",
            r.rank or "-",
            "-" if r.best_score is None else f"{r.best_score:.3f}",
            r.label.city,
            r.label.q,
        )
    hits = sum(r.rank is not None for r in results)
    mrr = sum(1 / r.rank for r in results if r.rank) / len(results)
    logger.info("hit rate@%d: %.2f (%d/%d)", k, hits / len(results), hits, len(results))
    logger.info("MRR@%d: %.3f", k, mrr)


async def run(settings: Settings) -> int:
    labels = read_labels(LABELS_FILE)
    if not labels:
        logger.error("no label in %s", LABELS_FILE)
        return 1
    chunks = [
        c
        for city, text in read_guides(settings.guides_dir).items()
        for c in chunk(city, text, settings.chunk_size, settings.chunk_overlap)
    ]
    if errors := find_label_errors(labels, chunks):
        for label in errors:
            logger.error(
                "label error: no %r chunk contains %r (%s)",
                label.city, label.must_contain, label.q,
            )
        return 1

    client = Mistral(api_key=settings.mistral_api_key.get_secret_value())
    embedder = MistralEmbedder(
        client=client, model=settings.embed_model, batch_size=settings.embed_batch_size
    )
    index = await load_or_build(settings.guides_dir, settings.index_dir, embedder, settings)
    results = await evaluate(labels, Retriever(index, embedder), settings.rag_top_k)
    log_report(results, settings.rag_top_k)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # one INFO line per request otherwise
    try:
        return asyncio.run(run(get_settings()))
    except TripError as exc:
        logger.error("evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
