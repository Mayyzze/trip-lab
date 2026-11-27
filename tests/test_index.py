from pathlib import Path

import numpy as np
import pytest

from trip.config import Settings
from trip.models import GuidesNotFound
from trip.rag import METADATA_FILE, VECTORS_FILE, Index, build_index, load_or_build

from .fakes import FakeEmbedder

DOCS = {
    "porto": "== Do ==\nTaste port wine in a cellar.",
    "lisbon": "== See ==\nVisit the Oceanarium.\n== Do ==\nRide tram 28.",
}


class UnnormalisedEmbedder:
    """Long vectors, and a zero one for texts about nothing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] if "nothing" in t else [3.0, 4.0] for t in texts]


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "mistral_api_key": "test-key",
        "embed_model": "mistral-embed",
        "chunk_size": 800,
        "chunk_overlap": 100,
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[arg-type]


@pytest.fixture
def guides_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "guides"
    directory.mkdir()
    for slug, text in DOCS.items():
        (directory / f"{slug}.txt").write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    return tmp_path / "index"


async def build(embedder: FakeEmbedder) -> Index:
    return await build_index(DOCS, embedder, model="mistral-embed", size=800, overlap=100)


async def test_build_embeds_every_chunk_in_one_call():
    embedder = FakeEmbedder()

    index = await build(embedder)

    assert [c.id for c in index.chunks] == ["lisbon#0", "lisbon#1", "porto#0"]
    assert embedder.calls == [[c.text for c in index.chunks]]
    assert index.vectors.shape == (3, 1024)
    assert index.vectors.dtype == np.float32
    assert sorted(index.manifest["guides"]) == ["lisbon", "porto"]


async def test_vectors_are_normalised_by_the_index():
    docs = {"lisbon": "== See ==\nCastle.\n== Do ==\nnothing"}

    index = await build_index(docs, UnnormalisedEmbedder(), model="m", size=800, overlap=100)

    np.testing.assert_allclose(index.vectors, [[0.6, 0.8], [0.0, 0.0]])
    assert index.vectors.dtype == np.float32


async def test_save_and_load_round_trip(index_dir: Path):
    index = await build(FakeEmbedder())

    index.save(index_dir)
    loaded = Index.load(index_dir)

    np.testing.assert_array_equal(loaded.vectors, index.vectors)
    assert loaded.vectors.dtype == np.float32
    assert loaded.chunks == index.chunks
    assert loaded.manifest == index.manifest


async def test_a_matching_manifest_makes_no_embedder_call(guides_dir: Path, index_dir: Path):
    first = await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())
    embedder = FakeEmbedder()

    second = await load_or_build(guides_dir, index_dir, embedder, make_settings())

    assert embedder.calls == []
    assert second.chunks == first.chunks


@pytest.mark.parametrize(
    "changed", [{"chunk_size": 500}, {"chunk_overlap": 50}, {"embed_model": "other-embed"}]
)
async def test_a_changed_setting_rebuilds(
    guides_dir: Path, index_dir: Path, changed: dict[str, object]
):
    await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())
    embedder = FakeEmbedder()

    index = await load_or_build(guides_dir, index_dir, embedder, make_settings(**changed))

    assert len(embedder.calls) == 1
    assert Index.load(index_dir).manifest == index.manifest  # the new index was saved


@pytest.mark.parametrize("filename", ["lisbon.txt", "faro.txt"])  # edited guide, new guide
async def test_a_changed_guide_rebuilds(guides_dir: Path, index_dir: Path, filename: str):
    await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())
    (guides_dir / filename).write_text("== Do ==\nSurf.", encoding="utf-8")
    embedder = FakeEmbedder()

    index = await load_or_build(guides_dir, index_dir, embedder, make_settings())

    assert len(embedder.calls) == 1
    assert any("Surf." in c.text for c in index.chunks)


async def test_a_removed_guide_rebuilds_without_its_chunks(guides_dir: Path, index_dir: Path):
    await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())
    (guides_dir / "porto.txt").unlink()

    index = await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())

    assert {c.city for c in index.chunks} == {"lisbon"}


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (METADATA_FILE, None),
        (METADATA_FILE, b"{truncated"),
        (METADATA_FILE, b"{}"),
        (VECTORS_FILE, None),
        (VECTORS_FILE, b""),
        (VECTORS_FILE, b"not a zip"),
        (VECTORS_FILE, b"PK\x03\x04 truncated zip"),
    ],
)
async def test_a_damaged_index_is_rebuilt(
    guides_dir: Path, index_dir: Path, filename: str, content: bytes | None
):
    await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())
    if content is None:
        (index_dir / filename).unlink()
    else:
        (index_dir / filename).write_bytes(content)
    embedder = FakeEmbedder()

    index = await load_or_build(guides_dir, index_dir, embedder, make_settings())

    assert len(embedder.calls) == 1
    assert len(Index.load(index_dir).chunks) == len(index.chunks)


async def test_no_guide_is_an_error_and_keeps_the_saved_index(guides_dir: Path, index_dir: Path):
    await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())
    for guide in guides_dir.glob("*.txt"):
        guide.unlink()

    with pytest.raises(GuidesNotFound):
        await load_or_build(guides_dir, index_dir, FakeEmbedder(), make_settings())

    assert Index.load(index_dir).chunks
