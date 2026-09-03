"""M1 data-pipeline tests against the prepared jsonl files."""

from pathlib import Path

import pytest

from ttsafety.data import load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_FILES = {
    "train": DATA_DIR / "harmful_train.jsonl",
    "val": DATA_DIR / "harmful_val.jsonl",
    "test": DATA_DIR / "harmful_test.jsonl",
}
HARMLESS_FILE = DATA_DIR / "harmless.jsonl"

_all_files = list(SPLIT_FILES.values()) + [HARMLESS_FILE]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _all_files),
    reason="prepared data files absent — run `uv run python scripts/prepare_data.py` first",
)


@pytest.fixture(scope="module")
def splits():
    return {name: load_jsonl(p) for name, p in SPLIT_FILES.items()}


def test_counts(splits):
    assert len(splits["train"]) == 256
    assert len(splits["val"]) == 64
    assert len(splits["test"]) > 0
    harmless = load_jsonl(HARMLESS_FILE)
    assert len(harmless) == 320


def test_schema(splits):
    for rows in splits.values():
        for row in rows:
            assert set(row.keys()) == {"instruction"}
            assert isinstance(row["instruction"], str) and row["instruction"].strip()
    for row in load_jsonl(HARMLESS_FILE):
        assert set(row.keys()) == {"instruction"}


def test_no_overlap_between_harmful_splits(splits):
    train = {r["instruction"] for r in splits["train"]}
    val = {r["instruction"] for r in splits["val"]}
    test = {r["instruction"] for r in splits["test"]}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_harmless_word_lengths():
    harmless = load_jsonl(HARMLESS_FILE)
    assert len({r["instruction"] for r in harmless}) == 320  # deduped
    for row in harmless:
        n = len(row["instruction"].split())
        assert 5 <= n <= 40
