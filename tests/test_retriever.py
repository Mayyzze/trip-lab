import numpy as np
import pytest

from trip.rag import Chunk, Index, Retriever, l2_normalise


class QueryEmbedder:
    """Every query points along the first axis, deliberately not normalised."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[5.0, 0.0] for _ in texts]


def make_index(*rows: tuple[str, float]) -> Index:
    """One chunk per (city, cosine similarity with the query)."""
    chunks = [Chunk(f"{city}#{i}", city, "Do", "text") for i, (city, _) in enumerate(rows)]
    vectors = l2_normalise([[score, np.sqrt(1 - score**2)] for _, score in rows])
    return Index(vectors, chunks, manifest={})


# The four best chunks overall are all in Porto.
INDEX = make_index(
    ("lisbon", 0.1),
    ("porto", 0.9),
    ("porto", 0.8),
    ("lisbon", 0.3),
    ("porto", 0.95),
    ("porto", 0.7),
    ("lisbon", 0.2),
)


async def test_the_city_filter_applies_before_the_top_k():
    hits = await Retriever(INDEX, QueryEmbedder()).search("rain", k=4, city="lisbon")

    assert [h.chunk.id for h in hits] == ["lisbon#3", "lisbon#6", "lisbon#0"]


async def test_hits_are_the_best_k_sorted_by_cosine_similarity():
    hits = await Retriever(INDEX, QueryEmbedder()).search("rain", k=3)

    assert [h.chunk.id for h in hits] == ["porto#4", "porto#1", "porto#2"]
    assert [h.score for h in hits] == pytest.approx([0.95, 0.9, 0.8])
    assert all(type(h.score) is float for h in hits)


async def test_k_larger_than_the_candidates_returns_them_all():
    hits = await Retriever(INDEX, QueryEmbedder()).search("rain", k=100, city="porto")

    assert [h.chunk.id for h in hits] == ["porto#4", "porto#1", "porto#2", "porto#5"]


async def test_the_query_is_embedded_once():
    embedder = QueryEmbedder()

    await Retriever(INDEX, embedder).search("rain", city="porto")

    assert embedder.calls == [["rain"]]


@pytest.mark.parametrize("city", ["faro", "Lisbon", ""])  # no guide, not a slug, empty
async def test_a_city_without_a_guide_gives_nothing_and_costs_no_call(city: str):
    embedder = QueryEmbedder()

    assert await Retriever(INDEX, embedder).search("rain", city=city) == []
    assert embedder.calls == []


@pytest.mark.parametrize("k", [0, -1])
async def test_k_must_be_positive(k: int):
    with pytest.raises(ValueError):
        await Retriever(INDEX, QueryEmbedder()).search("rain", k=k)
