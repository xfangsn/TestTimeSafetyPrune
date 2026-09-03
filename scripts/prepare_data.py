"""Prepare datasets (M1): AdvBench splits + Alpaca harmless sample, plus stats."""

import json
from pathlib import Path

from ttsafety.data import length_stats, prepare_all
from ttsafety.models import env_info

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    print("Preparing datasets ...")
    splits = prepare_all(DATA_DIR)

    stats = {"env": env_info(), "splits": {}}
    for name, rows in splits.items():
        stats["splits"][name] = length_stats(rows)
        s = stats["splits"][name]
        print(
            f"  {name}: {s['count']} rows, words mean={s['words_mean']:.1f} "
            f"median={s['words_median']} min={s['words_min']} max={s['words_max']}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "data_stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Stats saved to {out}")


if __name__ == "__main__":
    main()
