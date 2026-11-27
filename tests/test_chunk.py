from itertools import pairwise

import pytest

from trip.rag import Chunk, chunk


def words(prefix: str, n: int) -> str:
    """n distinct words, so that any repeated text between two chunks is a real overlap."""
    return " ".join(f"{prefix}{i:03d}" for i in range(n))


def body(c: Chunk) -> str:
    """Chunk text without its "<City> — <Section>" header line."""
    return c.text.split("\n", 1)[1]


def shared_overlap(previous: str, current: str) -> str:
    """Longest end of `previous` that is also the start of `current`."""
    for length in range(min(len(previous), len(current)), 0, -1):
        if previous.endswith(current[:length]):
            return current[:length]
    return ""


def paragraphs(prefix: str, count: int) -> str:
    return "\n".join(words(f"{prefix}{p}w", 8) for p in range(count))


GUIDE = f"""{words("lead", 30)}


== See ==
{paragraphs("see", 6)}

=== Museums ===
{paragraphs("museum", 4)}


== Do ==
{paragraphs("do", 6)}
"""


@pytest.mark.parametrize(("size", "overlap"), [(800, 100), (200, 40), (60, 20), (60, 0), (10, 9)])
def test_no_chunk_body_exceeds_size(size: int, overlap: int):
    chunks = chunk("lisbon", GUIDE, size=size, overlap=overlap)

    assert chunks
    for c in chunks:
        assert 0 < len(body(c)) <= size


def test_text_is_prefixed_with_city_and_section():
    [c] = chunk("lisbon", "== Do ==\nRide tram 28.")

    assert c == Chunk(id="lisbon#0", city="lisbon", section="Do", text="Lisbon — Do\nRide tram 28.")


def test_lead_section_has_an_empty_name():
    [c] = chunk("lisbon", "Capital of Portugal.")

    assert c.section == ""
    assert c.text == "Lisbon\nCapital of Portugal."


def test_hyphenated_slug_is_readable_in_the_prefix():
    [c] = chunk("san-sebastian", "== Eat ==\nPintxos.")

    assert c.city == "san-sebastian"
    assert c.text.startswith("San Sebastian — Eat\n")


def test_a_chunk_never_crosses_a_section_boundary():
    chunks = chunk("lisbon", GUIDE, size=200, overlap=40)

    assert [c.section for c in chunks] == sorted(
        (c.section for c in chunks), key=["", "See", "Do"].index
    )
    for c in chunks:
        expected = {"": ("lead",), "See": ("see", "museum", "Museums"), "Do": ("do",)}[c.section]
        assert all(word.startswith(expected) for word in body(c).split())


def test_subsection_keeps_the_parent_section_and_its_title():
    text = "== See ==\nCastles.\n=== Museums ===\nTile museum."

    first, second = chunk("lisbon", text)

    assert (first.section, body(first)) == ("See", "Castles.")
    assert (second.section, body(second)) == ("See", "Museums\nTile museum.")


def test_short_paragraphs_are_packed_together():
    [c] = chunk("lisbon", "== Do ==\nSail.\nSurf.\n\nSing.")

    assert body(c) == "Sail.\nSurf.\nSing."


def test_whole_paragraphs_are_not_split_when_they_fit():
    lines = [words(f"p{i}w", 8) for i in range(6)]  # 47 characters each

    chunks = chunk("lisbon", "\n".join(lines), size=120, overlap=20)

    for line in lines:
        assert any(line in body(c) for c in chunks)


def test_consecutive_chunks_of_a_section_overlap_on_a_word_boundary():
    chunks = chunk("lisbon", f"== Do ==\n{paragraphs('do', 10)}", size=150, overlap=30)

    assert len(chunks) > 2
    for previous, current in pairwise(chunks):
        shared = shared_overlap(body(previous), body(current))
        assert 0 < len(shared) <= 30
        assert body(previous)[-len(shared) - 1].isspace()  # the overlap starts a word


def test_zero_overlap_repeats_nothing():
    text = paragraphs("do", 10)

    chunks = chunk("lisbon", text, size=150, overlap=0)

    assert " ".join(body(c) for c in chunks).split() == text.split()


def test_long_paragraph_is_split_on_sentence_boundaries():
    sentences = [f"{words(f's{i}w', 6)}." for i in range(8)]  # 42 characters each

    chunks = chunk("lisbon", " ".join(sentences), size=100, overlap=0)

    assert len(chunks) > 1
    assert all(body(c).endswith(".") for c in chunks)
    assert " ".join(body(c) for c in chunks) == " ".join(sentences)


def test_a_sentence_longer_than_size_is_hard_cut_without_losing_text():
    token = "".join(f"{i:04d}" for i in range(100))  # 400 characters, no boundary at all

    chunks = chunk("lisbon", token, size=150, overlap=0)

    assert [len(body(c)) for c in chunks] == [150, 150, 100]
    assert "".join(body(c) for c in chunks) == token


def test_a_hard_cut_never_produces_an_empty_chunk():
    chunks = chunk("lisbon", "ab cd", size=1, overlap=0)

    assert [body(c) for c in chunks] == ["a", "b", "c", "d"]


def test_every_word_survives_chunking():
    text = paragraphs("do", 10)

    chunks = chunk("lisbon", text, size=100, overlap=25)

    assert {word for c in chunks for word in body(c).split()} == set(text.split())


def test_empty_sections_are_skipped():
    text = "== See ==\n\n=== Museums ===\n\n== Do ==\nSail.\n\n== Eat ==\n"

    assert [c.section for c in chunk("lisbon", text)] == ["Do"]


@pytest.mark.parametrize("text", ["", "  \n\n ", "== See ==\n"])
def test_text_without_content_gives_no_chunk(text: str):
    assert chunk("lisbon", text) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (100, 100), (100, 101), (100, -1)])
def test_invalid_parameters_are_rejected(size: int, overlap: int):
    with pytest.raises(ValueError):
        chunk("lisbon", GUIDE, size=size, overlap=overlap)


def test_ids_are_numbered_across_sections_and_stable():
    first = chunk("lisbon", GUIDE, size=200, overlap=40)
    second = chunk("lisbon", GUIDE, size=200, overlap=40)

    assert first == second
    assert [c.id for c in first] == [f"lisbon#{i}" for i in range(len(first))]
