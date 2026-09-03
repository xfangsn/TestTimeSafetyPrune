"""N1: per-sample minimal flip-alpha search on harmful_train (bisection).

Chunkable: `--start 0 --end 64` processes that slice and merges records into
data/flip_alphas.jsonl. `--finalize` aggregates stats, writes
results/flip_alpha.json + flip_alpha_hist.png, and prints boundary samples.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ttsafety.data import load_jsonl
from ttsafety.generate import generate_texts
from ttsafety.judge import is_refusal
from ttsafety.models import env_info, load_model
from ttsafety.steer import steer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RECORDS_PATH = DATA_DIR / "flip_alphas.jsonl"
MODEL_TAG = "llama32_3b_instruct"
ALPHA_MAX = 12.0
TOL = 0.25
MAX_NEW_TOKENS = 128


def load_records() -> dict[int, dict]:
    if not RECORDS_PATH.exists():
        return {}
    return {r["idx"]: r for r in load_jsonl(RECORDS_PATH)}


def save_records(records: dict[int, dict]) -> None:
    with RECORDS_PATH.open("w", encoding="utf-8") as f:
        for idx in sorted(records):
            f.write(json.dumps(records[idx], ensure_ascii=False) + "\n")


def search_sample(model, tokenizer, vec, layer, instruction):
    """Bisect minimal |alpha| flipping this sample; returns (alpha*, output)."""
    def probe(a):
        with steer(model, vec, layer=layer, alpha=-a):
            out = generate_texts(model, tokenizer, [instruction],
                                 max_new_tokens=MAX_NEW_TOKENS, batch_size=1)[0]
        return (not is_refusal(out)), out

    flipped, out = probe(ALPHA_MAX)
    if not flipped:
        return None, out  # unflippable within [0, ALPHA_MAX]
    lo, hi = 0.0, ALPHA_MAX
    best = out
    while hi - lo > TOL:
        mid = (lo + hi) / 2
        flipped, out = probe(mid)
        if flipped:
            hi, best = mid, out
        else:
            lo = mid
    return hi, best


def run_range(start, end, layer):
    model, tokenizer = load_model()
    directions = torch.load(
        DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt", weights_only=True)
    vec = directions[layer]
    instructions = [r["instruction"]
                    for r in load_jsonl(DATA_DIR / "harmful_train.jsonl")]
    chunk = instructions[start:end]

    records = load_records()
    todo = [(start + i, s) for i, s in enumerate(chunk) if start + i not in records]
    if not todo:
        print("nothing to do for this range")
        return

    # batched alpha=0 baseline for the chunk's remaining samples
    print(f"baseline pass on {len(todo)} samples ...")
    base_outs = generate_texts(model, tokenizer, [s for _, s in todo],
                               MAX_NEW_TOKENS)
    refusers = []
    for (idx, s), out in zip(todo, base_outs):
        if is_refusal(out):
            refusers.append((idx, s))
        else:
            records[idx] = {"idx": idx, "instruction": s,
                            "refused_baseline": False,
                            "alpha_star": None, "output": out}
    print(f"refused at baseline: {len(refusers)}/{len(todo)}; bisecting ...")

    for n, (idx, s) in enumerate(refusers):
        alpha_star, out = search_sample(model, tokenizer, vec, layer, s)
        records[idx] = {"idx": idx, "instruction": s, "refused_baseline": True,
                        "alpha_star": alpha_star, "output": out}
        if (n + 1) % 10 == 0:
            save_records(records)
            print(f"  {n + 1}/{len(refusers)} done")
    save_records(records)
    print(f"saved {len(records)} total records to {RECORDS_PATH}")


def finalize():
    records = load_records()
    n = len(records)
    refused = [r for r in records.values() if r["refused_baseline"]]
    flipped = [r for r in refused if r["alpha_star"] is not None]
    unflippable = [r for r in refused if r["alpha_star"] is None]
    alphas = sorted(r["alpha_star"] for r in flipped)

    def pct(p):
        return alphas[min(len(alphas) - 1, int(p * len(alphas)))] if alphas else None

    stats = {
        "config": {"model": MODEL_TAG, "alpha_max": ALPHA_MAX, "tol": TOL,
                   "max_new_tokens": MAX_NEW_TOKENS,
                   "steer_layer": 8},
        "env": env_info(),
        "n_total": n,
        "n_refused_baseline": len(refused),
        "n_not_refused_baseline": n - len(refused),
        "n_flipped": len(flipped),
        "n_unflippable": len(unflippable),
        "flippable_rate_of_all": len(flipped) / n if n else 0,
        "flippable_rate_of_refused": len(flipped) / len(refused) if refused else 0,
        "alpha_star": {
            "mean": sum(alphas) / len(alphas) if alphas else None,
            "median": pct(0.5),
            "q25": pct(0.25),
            "q75": pct(0.75),
        },
    }
    out_path = RESULTS_DIR / "flip_alpha.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(alphas, bins=[i * TOL for i in range(int(ALPHA_MAX / TOL) + 1)],
            color="tab:blue", alpha=0.8)
    ax.bar([ALPHA_MAX + 0.5], [len(unflippable)], width=0.9,
           color="tab:red", label=f"unflippable ({len(unflippable)})")
    ax.set_xlabel("alpha* (minimal |alpha| flipping refusal at L8)")
    ax.set_ylabel("count")
    ax.set_title(f"Flip-alpha distribution (n_flipped={len(flipped)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "flip_alpha_hist.png", dpi=150)
    print(f"saved {out_path} and results/flip_alpha_hist.png")

    print("\n== 5 smallest alpha* (flipped) ==")
    for r in sorted(flipped, key=lambda r: r["alpha_star"])[:5]:
        print(f"  a*={r['alpha_star']:.2f} {r['instruction'][:60]!r}\n"
              f"    -> {r['output'][:180]!r}")
    print("\n== 5 unflippable (output at alpha=12) ==")
    for r in unflippable[:5]:
        print(f"  {r['instruction'][:60]!r}\n    -> {r['output'][:180]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=256)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
    else:
        run_range(args.start, args.end, args.layer)


if __name__ == "__main__":
    main()
