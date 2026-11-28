"""The hand-labelled questions of the extraction eval, read by the tests and the eval script."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tag = Literal[
    "relative_date",
    "convention",
    "non_english",
    "no_date",
    "ambiguous_city",
    "two_cities",
    "beyond_horizon",
    "typo",
]


class DatasetItem(BaseModel):
    """A question, the date it is asked and the expected extraction. See evals/LABELLING.md."""

    # A misspelt key must fail: ignored, it would leave an optional label silently null.
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    q: str = Field(min_length=1)
    today: date
    city: str | None  # no default: an unlabelled field is an error, not a null label
    start: date | None
    end: date | None
    topic: Literal["weather", "activities", "both"]
    country: str | None = None
    region: str | None = None
    lang: Literal["en", "fr", "it", "es"]
    tags: frozenset[Tag]


def load_dataset(path: Path) -> list[DatasetItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [DatasetItem.model_validate_json(line) for line in lines if line.strip()]
