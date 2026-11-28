"""The dataset is hand-written: these tests catch labelling slips, not wrong judgement."""

from collections import Counter
from datetime import timedelta

import pytest

from trip.config import PROJECT_ROOT, Settings
from trip.dataset import DatasetItem, Tag, load_dataset

ITEMS = load_dataset(PROJECT_ROOT / "evals" / "dataset.jsonl")
# The field default, read without building Settings: that would require an API key.
HORIZON_DAYS: int = Settings.model_fields["forecast_horizon_days"].default
MIN_ITEMS_PER_TAG: dict[Tag, int] = {
    "relative_date": 8,
    "convention": 2,
    "non_english": 6,
    "no_date": 4,
    "ambiguous_city": 2,
    "two_cities": 2,
    "beyond_horizon": 2,
    "typo": 2,
}
MIN_ACTIVITIES_OR_BOTH = 3


def by_id(item: DatasetItem) -> str:
    return item.id


def test_there_are_30_items_with_unique_ids():
    assert len(ITEMS) == 30
    assert len({item.id for item in ITEMS}) == 30


@pytest.mark.parametrize("item", ITEMS, ids=by_id)
def test_dates_come_in_ordered_pairs_from_today_on(item: DatasetItem):
    if item.start is None or item.end is None:
        assert item.start is None and item.end is None
    else:
        assert item.today <= item.start <= item.end


@pytest.mark.parametrize(("tag", "minimum"), MIN_ITEMS_PER_TAG.items())
def test_each_tag_reaches_its_coverage_target(tag: Tag, minimum: int):
    assert sum(tag in item.tags for item in ITEMS) >= minimum


def test_enough_questions_ask_for_activities():
    topics = Counter(item.topic for item in ITEMS)

    assert topics["activities"] + topics["both"] >= MIN_ACTIVITIES_OR_BOTH


def test_today_varies_over_every_weekday():
    assert {item.today.weekday() for item in ITEMS} == set(range(7))


@pytest.mark.parametrize("item", ITEMS, ids=by_id)
def test_tags_agree_with_the_labels(item: DatasetItem):
    assert ("non_english" in item.tags) == (item.lang != "en")
    if "no_date" in item.tags:
        assert item.start is None and item.end is None
    if "two_cities" in item.tags:
        assert item.city is None
    if "beyond_horizon" in item.tags:
        last_forecast_day = item.today + timedelta(days=HORIZON_DAYS - 1)
        assert item.start is not None and item.start > last_forecast_day
