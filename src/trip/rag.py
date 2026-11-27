"""Retrieval over travel guides. Knows nothing about any SDK."""

import hashlib
import json
import logging
import re
import unicodedata
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from trip.config import Settings
from trip.models import GuidesNotFound
from trip.ports import Embedder

# "== See ==" or "=== Museums ===" (deeper levels too), as written by exsectionformat=wiki.
HEADER = re.compile(r"^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CITATION = re.compile(r"\[([a-z0-9_-]+#\d+)\]")  # a chunk ID in brackets, e.g. [lisbon#12]
VECTORS_FILE = "index.npz"
METADATA_FILE = "index.json"  # chunks and manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    id: str  # f"{city}#{i}", deterministic for a given text and parameters
    city: str
    section: str  # top-level title, e.g. "Do" or "See"; "" for the lead section
    text: str  # text that gets embedded, prefixed with "<City> — <Section>\n"


def slugify(city: str) -> str:
    """The guide slug of a city name, lowercase ASCII: "Kraków" -> "krakow"."""
    ascii_name = unicodedata.normalize("NFKD", city.strip()).encode("ascii", "ignore").decode()
    return ascii_name.lower().replace(" ", "-")


def check_citations(answer: str, provided_ids: set[str]) -> list[str]:
    """The provided chunk IDs that `answer` cites, in order of appearance, without repeats."""
    return [i for i in dict.fromkeys(CITATION.findall(answer)) if i in provided_ids]


def chunk(city: str, text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Cut a guide into chunks of at most `size` characters, header line excluded.

    `city` is the guide slug. A chunk never crosses a section or subsection boundary.
    Consecutive chunks of a block share up to `overlap` characters.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, {size}), got {overlap}")

    name = city.replace("-", " ").title()
    chunks: list[Chunk] = []
    for section, block in _blocks(text):
        header = f"{name} — {section}" if section else name
        for body in _pack(_units(block, size, overlap), size, overlap):
            chunks.append(Chunk(f"{city}#{len(chunks)}", city, section, f"{header}\n{body}"))
    return chunks


def _blocks(text: str) -> Iterator[tuple[str, str]]:
    """(section, text) for the lead and for each header, in document order.

    A subsection belongs to its parent section; its title becomes its first line.
    """
    # split() with two groups gives [lead, marks, title, body, marks, title, body, ...]
    parts = HEADER.split(text)
    yield "", parts[0]
    section = ""
    for marks, title, body in zip(parts[1::3], parts[2::3], parts[3::3], strict=True):
        if marks == "==":
            section = title
            yield section, body
        elif body.strip():  # a title alone is not content
            yield section, f"{title}\n{body}"


def _units(block: str, size: int, overlap: int) -> Iterator[tuple[str, str]]:
    """The pieces to pack, none longer than `size`, each with the separator preceding it.

    A piece is a whole paragraph (one line) if it fits, else a sentence, else a slice.
    """
    for paragraph in filter(None, map(str.strip, block.splitlines())):
        if len(paragraph) <= size:
            yield "\n", paragraph
            continue
        separator = "\n"
        for sentence in SENTENCE_END.split(paragraph):
            # Slices leave room for the overlap, so that hard cuts get one too.
            step = len(sentence) if len(sentence) <= size else size - overlap
            for start in range(0, len(sentence), step):
                yield separator, sentence[start : start + step]
                separator = ""  # the next slice continues the same sentence
            separator = " "


def _pack(units: Iterator[tuple[str, str]], size: int, overlap: int) -> Iterator[str]:
    """Greedily join pieces into bodies of at most `size` characters."""
    current = ""
    for separator, unit in units:
        if not current:
            current = unit
        elif len(current) + len(separator) + len(unit) <= size:
            current += separator + unit
        else:
            if body := current.strip():  # a hard-cut slice can be pure whitespace
                yield body
            # The overlap shrinks when the next piece leaves less than `overlap` characters.
            tail = _tail(current, min(overlap, size - len(separator) - len(unit)))
            current = tail + separator + unit if tail else unit
    if body := current.strip():
        yield body


def _tail(text: str, max_chars: int) -> str:
    """Longest end of `text`, at most `max_chars` long, that starts on a word boundary."""
    if max_chars <= 0:
        return ""
    # One extra character tells whether the cut falls between two words.
    window = text[-max_chars - 1 :]
    boundary = re.search(r"\s+", window)
    return window[boundary.end() :] if boundary else ""


def l2_normalise(vectors: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Vectors (last axis) scaled to unit length, as float32. A zero vector stays zero."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms == 0, 1, norms)


@dataclass(frozen=True, eq=False)  # comparing arrays with == is ambiguous
class Index:
    vectors: npt.NDArray[np.float32]  # shape (len(chunks), dimension), rows L2-normalised
    chunks: list[Chunk]
    manifest: dict[str, Any]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        metadata = path / METADATA_FILE
        # Removed first and written last: an interrupted save leaves no loadable index.
        metadata.unlink(missing_ok=True)
        np.savez(path / VECTORS_FILE, vectors=self.vectors)
        payload = {"manifest": self.manifest, "chunks": [asdict(c) for c in self.chunks]}
        metadata.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a saved index. Raises if a file is missing, damaged or inconsistent."""
        payload = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
        with np.load(path / VECTORS_FILE) as data:
            vectors = data["vectors"]
        chunks = [Chunk(**c) for c in payload["chunks"]]
        if vectors.dtype != np.float32 or vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError(f"vectors {vectors.dtype}{vectors.shape} for {len(chunks)} chunks")
        return cls(vectors, chunks, payload["manifest"])


def make_manifest(docs: dict[str, str], *, model: str, size: int, overlap: int) -> dict[str, Any]:
    """Everything the vectors depend on: an index is valid only while this is unchanged."""
    return {
        "embed_model": model,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "guides": {slug: hashlib.sha256(text.encode()).hexdigest() for slug, text in docs.items()},
    }


async def build_index(
    docs: dict[str, str], embedder: Embedder, *, model: str, size: int, overlap: int
) -> Index:
    """Chunk and embed `docs` (slug -> guide text). `model` is recorded in the manifest."""
    chunks = [c for city, text in sorted(docs.items()) for c in chunk(city, text, size, overlap)]
    vectors = np.asarray(await embedder.embed([c.text for c in chunks]), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(chunks):
        raise ValueError(f"embeddings of shape {vectors.shape} for {len(chunks)} chunks")
    manifest = make_manifest(docs, model=model, size=size, overlap=overlap)
    return Index(l2_normalise(vectors), chunks, manifest)


def read_guides(guides_dir: Path) -> dict[str, str]:
    """slug -> text for every guide in the directory."""
    docs = {f.stem: f.read_text(encoding="utf-8") for f in sorted(guides_dir.glob("*.txt"))}
    if not docs:
        raise GuidesNotFound(f"no guide in {guides_dir}, run scripts/fetch_guides.py")
    return docs


async def load_or_build(
    guides_dir: Path, index_dir: Path, embedder: Embedder, settings: Settings
) -> Index:
    """The saved index if its manifest matches the guides and settings, else a new saved one."""
    docs = read_guides(guides_dir)
    expected = make_manifest(
        docs, model=settings.embed_model, size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    try:
        index = Index.load(index_dir)
    except FileNotFoundError:
        logger.info("no index in %s, building it", index_dir)
    except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        logger.warning("unreadable index in %s, rebuilding it: %r", index_dir, exc)
    else:
        if index.manifest == expected:
            return index
        logger.info("index manifest changed, rebuilding it")

    index = await build_index(
        docs,
        embedder,
        model=settings.embed_model,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    index.save(index_dir)
    return index


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity with the query


class Retriever:
    def __init__(self, index: Index, embedder: Embedder) -> None:
        """`embedder` must use the model the index was built with."""
        self._index = index
        self._embedder = embedder
        self._cities = np.array([c.city for c in index.chunks])

    async def search(self, query: str, k: int = 4, city: str | None = None) -> list[Hit]:
        """The `k` chunks closest to `query`, best first, among those of `city` (a guide slug).

        Returns [] for a city without a guide; it never widens the search to other cities.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        # Filtered before the top-k selection, so other cities cannot crowd this one out.
        mask = np.ones(len(self._cities), dtype=bool) if city is None else self._cities == city
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []

        [vector] = await self._embedder.embed([query])
        scores = self._index.vectors[candidates] @ l2_normalise(vector)
        k = min(k, candidates.size)
        top = np.argpartition(scores, -k)[-k:]  # the k best, in no particular order
        top = top[np.argsort(scores[top])[::-1]]
        return [Hit(self._index.chunks[candidates[i]], float(scores[i])) for i in top]
