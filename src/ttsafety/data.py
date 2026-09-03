"""Dataset loading: AdvBench harmful splits + Alpaca harmless sample."""

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

ADV_BENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
SEED = 0
N_HARMFUL_TRAIN = 256
N_HARMFUL_VAL = 64
N_HARMLESS = 320
MIN_WORDS, MAX_WORDS = 5, 40


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_advbench(cache_dir: Path) -> list[str]:
    """Download harmful_behaviors.csv (cached); fail loudly on network errors."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "harmful_behaviors.csv"
    if not cache_path.exists():
        try:
            with urllib.request.urlopen(ADV_BENCH_URL, timeout=60) as resp:
                cache_path.write_bytes(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to download AdvBench from {ADV_BENCH_URL}: {e}. "
                "Provide a local copy at data/advbench/harmful_behaviors.csv."
            ) from e
    text = cache_path.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    # Header is "goal,target"; take the goal column.
    header = [c.strip().lower() for c in rows[0]]
    goal_idx = header.index("goal") if "goal" in header else 0
    return [r[goal_idx] for r in rows[1:] if r and r[goal_idx].strip()]


def split_harmful(behaviors: list[str]) -> dict[str, list[str]]:
    """Fixed-seed 0 split: 256 train / 64 val / rest test."""
    idx = list(range(len(behaviors)))
    random.Random(SEED).shuffle(idx)
    train = idx[:N_HARMFUL_TRAIN]
    val = idx[N_HARMFUL_TRAIN : N_HARMFUL_TRAIN + N_HARMFUL_VAL]
    test = idx[N_HARMFUL_TRAIN + N_HARMFUL_VAL :]
    return {
        "train": [behaviors[i] for i in train],
        "val": [behaviors[i] for i in val],
        "test": [behaviors[i] for i in test],
    }


def sample_harmless() -> list[str]:
    """320 short, input-free, deduped Alpaca instructions (seed 0)."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    seen, candidates = set(), []
    for ex in ds:
        if ex["input"].strip() != "":
            continue
        instr = ex["instruction"].strip()
        n_words = len(instr.split())
        if not (MIN_WORDS <= n_words <= MAX_WORDS):
            continue
        if instr in seen:
            continue
        seen.add(instr)
        candidates.append(instr)
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    if len(candidates) < N_HARMLESS:
        raise RuntimeError(
            f"Only {len(candidates)} harmless candidates, need {N_HARMLESS}."
        )
    return candidates[:N_HARMLESS]


def prepare_all(data_dir: Path) -> dict[str, list[str]]:
    """Run the full data pipeline and write jsonl files under data_dir."""
    behaviors = fetch_advbench(data_dir / "advbench")
    splits = split_harmful(behaviors)
    for name, rows in splits.items():
        _write_jsonl(
            [{"instruction": s} for s in rows], data_dir / f"harmful_{name}.jsonl"
        )
    harmless = sample_harmless()
    _write_jsonl([{"instruction": s} for s in harmless], data_dir / "harmless.jsonl")
    return {**splits, "harmless": harmless}


def length_stats(rows: list[str]) -> dict:
    words = sorted(len(r.split()) for r in rows)
    n = len(words)
    median = words[n // 2] if n % 2 else (words[n // 2 - 1] + words[n // 2]) / 2
    return {
        "count": n,
        "words_mean": sum(words) / n,
        "words_median": median,
        "words_min": words[0],
        "words_max": words[-1],
    }
